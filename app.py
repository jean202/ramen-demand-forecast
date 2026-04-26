import os
from pathlib import Path

import anthropic
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

OUTPUT_DIR = Path(__file__).parent / "DemandForecast" / "output"
HOLDOUT_START = "2023-07-01"
ACTUAL_END = "2023-12-01"
FUTURE_START = "2024-01-01"

st.set_page_config(page_title="라면류 수요예측", page_icon="🍜", layout="wide")


@st.cache_data
def load_data():
    forecast = pd.read_csv(OUTPUT_DIR / "prophet_forecast.csv", parse_dates=["ds"])
    holdout = pd.read_csv(OUTPUT_DIR / "prophet_holdout_comparison.csv", parse_dates=["ds"])
    metrics = pd.read_csv(OUTPUT_DIR / "prophet_metrics.csv")
    tuning = pd.read_csv(OUTPUT_DIR / "prophet_tuning_results.csv")
    return forecast, holdout, metrics, tuning


def get_anthropic_client() -> anthropic.Anthropic | None:
    api_key = st.secrets.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    return anthropic.Anthropic(api_key=api_key)


def stream_explanation(prompt: str):
    client = get_anthropic_client()
    if client is None:
        yield "API 키가 설정되지 않았습니다. `.streamlit/secrets.toml`에 `ANTHROPIC_API_KEY`를 입력하세요."
        return
    with client.messages.stream(
        model="claude-sonnet-4-6",
        max_tokens=512,
        system=(
            "너는 데이터 분석 결과를 명확하게 설명하는 전문가야. "
            "비전문가도 이해할 수 있도록 핵심만 간결하게 설명해. "
            "마크다운 헤더(#)는 쓰지 말고, 자연스러운 문단 형태로 작성해."
        ),
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for text in stream.text_stream:
            yield text


def ai_explain_button(key: str, prompt: str):
    if st.button(":material/auto_awesome: AI 해설 생성", key=key):
        with st.container(border=True):
            st.write_stream(stream_explanation(prompt))


forecast_df, holdout_df, metrics_df, tuning_df = load_data()

# ── 헤더 ─────────────────────────────────────────────────────────────────────
st.title("🍜 라면류 수요예측 대시보드")
st.caption("KT AIVLE 리테일 데이터 페스타 · 2021~2023 거래 데이터 기반 Prophet 수요예측")

# ── 상단 KPI ──────────────────────────────────────────────────────────────────
best = metrics_df[metrics_df["모델"] == "튜닝 Prophet"].iloc[0]
baseline = metrics_df[metrics_df["모델"] == "전년동월 기준선"].iloc[0]
default_m = metrics_df[metrics_df["모델"] == "기본 Prophet"].iloc[0]

c1, c2, c3, c4 = st.columns(4)
c1.metric("MAPE (튜닝 Prophet)", f"{best['MAPE(%)']:.2f}%")
c2.metric("RMSE (튜닝 Prophet)", f"{best['RMSE']:,.0f}")
c3.metric("기본 Prophet 대비 MAPE", f"{best['MAPE(%)']:.2f}%", f"-{default_m['MAPE(%)'] - best['MAPE(%)']:.2f}%p")
c4.metric("전년동월 기준선 대비 MAPE", f"{best['MAPE(%)']:.2f}%", f"-{baseline['MAPE(%)'] - best['MAPE(%)']:.2f}%p")

st.divider()

# ── 탭 ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    ":material/show_chart: 수요 예측",
    ":material/leaderboard: 모델 성능 비교",
    ":material/manage_search: Holdout 검증",
    ":material/tune: 튜닝 결과",
])

# ── 탭 1: 수요 예측 ───────────────────────────────────────────────────────────
with tab1:
    st.subheader(":material/show_chart: 라면류 월별 판매수량 예측 (2021~2026)")

    hist = forecast_df[forecast_df["ds"] <= ACTUAL_END].copy()
    future = forecast_df[forecast_df["ds"] >= FUTURE_START].copy()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pd.concat([hist["ds"], hist["ds"][::-1]]),
        y=pd.concat([hist["yhat_upper"], hist["yhat_lower"][::-1]]),
        fill="toself", fillcolor="rgba(196, 61, 92, 0.12)",
        line=dict(color="rgba(0,0,0,0)"), name="신뢰구간 (학습·검증)",
    ))
    fig.add_trace(go.Scatter(
        x=pd.concat([future["ds"], future["ds"][::-1]]),
        y=pd.concat([future["yhat_upper"], future["yhat_lower"][::-1]]),
        fill="toself", fillcolor="rgba(100, 120, 200, 0.12)",
        line=dict(color="rgba(0,0,0,0)"), name="신뢰구간 (미래 예측)",
    ))
    fig.add_trace(go.Scatter(
        x=hist["ds"], y=hist["yhat"], mode="lines",
        line=dict(color="#c43d5c", width=2), name="Prophet 예측 (학습·검증)",
    ))
    fig.add_trace(go.Scatter(
        x=future["ds"], y=future["yhat"], mode="lines",
        line=dict(color="#5b6abf", width=2, dash="dot"), name="Prophet 예측 (미래)",
    ))
    fig.add_trace(go.Scatter(
        x=holdout_df["ds"], y=holdout_df["y"], mode="markers+lines",
        marker=dict(color="#2f5d50", size=8), line=dict(color="#2f5d50", width=1.5),
        name="실제값 (Holdout)",
    ))
    fig.add_vrect(
        x0=pd.Timestamp(HOLDOUT_START), x1=pd.Timestamp(ACTUAL_END),
        fillcolor="#d9edf7", opacity=0.25, layer="below", line_width=0,
        annotation_text="Holdout", annotation_position="top left",
    )
    fig.add_vline(
        x=pd.Timestamp(FUTURE_START).timestamp() * 1000,
        line_dash="dash", line_color="gray", opacity=0.5,
        annotation_text="예측 시작", annotation_position="top right",
    )
    fig.update_layout(
        xaxis_title="날짜", yaxis_title="판매수량",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified", height=480, margin=dict(t=60),
    )
    st.plotly_chart(fig, use_container_width=True)

    annual = (
        forecast_df[forecast_df["ds"] >= FUTURE_START]
        .assign(연도=lambda df: df["ds"].dt.year)
        .groupby("연도")["yhat"].sum().round(0).reset_index()
        .rename(columns={"yhat": "예측 판매수량 합계"})
    )
    annual_text = "\n".join(
        f"  {int(r['연도'])}년: {r['예측 판매수량 합계']:,.0f}" for _, r in annual.iterrows()
    )

    st.subheader(":material/calendar_month: 연간 예측 합계")
    annual["예측 판매수량 합계"] = annual["예측 판매수량 합계"].map("{:,.0f}".format)
    st.dataframe(annual, use_container_width=True, hide_index=True)

    ai_explain_button("explain_forecast", f"""
라면류 월별 수요예측 차트에 대한 설명을 작성해줘.

데이터 요약:
- 학습 기간: 2021-01 ~ 2023-06 (30개월)
- Holdout(검증) 기간: 2023-07 ~ 2023-12 (6개월)
- 미래 예측 기간: 2024-01 ~ 2026-12
- 연간 예측 합계:
{annual_text}

차트에는 다음 요소가 있어:
1. 붉은 실선: Prophet 예측값 (학습·검증 구간)
2. 파란 점선: Prophet 예측값 (미래 구간)
3. 초록 점+선: Holdout 실제값
4. 연한 파란 음영: Holdout 구간 표시
5. 회색 점선: 미래 예측 시작 기준선
6. 신뢰구간 음영

이 차트가 왜 이런 구조로 만들어졌는지, 그리고 차트 모양에서 읽을 수 있는 인사이트(계절성, 트렌드 방향 등)를 3~4문장으로 설명해줘.
""")

# ── 탭 2: 모델 성능 비교 ──────────────────────────────────────────────────────
with tab2:
    st.subheader(":material/leaderboard: 모델별 Holdout 성능 비교 (2023 H2)")

    col_chart, col_table = st.columns([2, 1])
    with col_chart:
        colors = ["#c43d5c", "#888888", "#cccccc"]
        fig2 = go.Figure()
        for i, metric in enumerate(["MAPE(%)", "RMSE", "MAE"]):
            fig2.add_trace(go.Bar(
                name=metric, x=metrics_df["모델"], y=metrics_df[metric],
                marker_color=colors, text=metrics_df[metric].round(2),
                textposition="outside", visible=(i == 0),
            ))
        fig2.update_layout(
            updatemenus=[dict(
                type="buttons", direction="left",
                buttons=[
                    dict(label="MAPE (%)", method="update", args=[{"visible": [True, False, False]}]),
                    dict(label="RMSE", method="update", args=[{"visible": [False, True, False]}]),
                    dict(label="MAE", method="update", args=[{"visible": [False, False, True]}]),
                ],
                x=0, y=1.15, xanchor="left",
            )],
            yaxis_title="값", height=400, margin=dict(t=80), showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col_table:
        st.markdown("**성능 수치**")
        styled = metrics_df.style.highlight_min(subset=["RMSE", "MAE", "MAPE(%)"], color="#d4edda", axis=0)
        st.dataframe(styled, use_container_width=True, hide_index=True)
        st.markdown("**개선율 (기본 Prophet 대비)**")
        improvement = round((default_m["MAPE(%)"] - best["MAPE(%)"]) / default_m["MAPE(%)"] * 100, 1)
        st.metric("MAPE 상대 개선율", f"{improvement}%")

    metrics_text = metrics_df.to_string(index=False)
    ai_explain_button("explain_metrics", f"""
라면류 수요예측 모델 3개의 성능 비교 결과야.

{metrics_text}

- 튜닝 Prophet: Prophet 파라미터를 holdout 검증 기반으로 최적화한 모델
- 기본 Prophet: 튜닝 없이 기본값으로 실행한 모델
- 전년동월 기준선: 작년 같은 달 값을 그대로 예측값으로 쓰는 단순 baseline

이 세 모델의 성능 차이가 왜 발생하는지, 그리고 이 결과에서 어떤 의미를 읽을 수 있는지 3~4문장으로 설명해줘.
""")

# ── 탭 3: Holdout 검증 ───────────────────────────────────────────────────────
with tab3:
    st.subheader(":material/manage_search: Holdout 구간 실제값 vs 예측값 (2023년 7~12월)")

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=holdout_df["ds"], y=holdout_df["yhat_lower"],
        mode="lines", line=dict(color="rgba(0,0,0,0)"), showlegend=False, name="하한",
    ))
    fig3.add_trace(go.Scatter(
        x=holdout_df["ds"], y=holdout_df["yhat_upper"],
        mode="lines", line=dict(color="rgba(0,0,0,0)"),
        fill="tonexty", fillcolor="rgba(196, 61, 92, 0.15)", name="신뢰구간",
    ))
    fig3.add_trace(go.Scatter(
        x=holdout_df["ds"], y=holdout_df["y"], mode="markers+lines",
        marker=dict(color="#2f5d50", size=10), line=dict(color="#2f5d50", width=2), name="실제값",
    ))
    fig3.add_trace(go.Scatter(
        x=holdout_df["ds"], y=holdout_df["튜닝Prophet"], mode="markers+lines",
        marker=dict(color="#c43d5c", size=8, symbol="diamond"),
        line=dict(color="#c43d5c", width=2), name="튜닝 Prophet",
    ))
    fig3.add_trace(go.Scatter(
        x=holdout_df["ds"], y=holdout_df["기본Prophet"], mode="markers+lines",
        marker=dict(color="#aaaaaa", size=6, symbol="square"),
        line=dict(color="#aaaaaa", width=1.5, dash="dot"), name="기본 Prophet",
    ))
    fig3.update_layout(
        xaxis_title="날짜", yaxis_title="판매수량", hovermode="x unified", height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(fig3, use_container_width=True)

    display_df = holdout_df[["ds", "y", "튜닝Prophet", "기본Prophet"]].copy()
    display_df["ds"] = display_df["ds"].dt.strftime("%Y-%m")
    display_df["오차(튜닝)"] = (display_df["y"] - display_df["튜닝Prophet"]).round(2)
    display_df.columns = ["월", "실제값", "튜닝 Prophet", "기본 Prophet", "오차(튜닝)"]
    st.markdown("**월별 상세**")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    holdout_text = display_df.to_string(index=False)
    ai_explain_button("explain_holdout", f"""
라면류 수요예측 모델의 Holdout 검증 결과야. 2023년 하반기(7~12월)를 학습에서 제외하고 시험지처럼 사용했어.

{holdout_text}

월별 실제값과 예측값의 차이 패턴, 어느 달에 잘 맞고 어느 달에 벗어났는지, 그리고 이런 오차 패턴이 왜 생겼을지 3~4문장으로 설명해줘.
""")

# ── 탭 4: 튜닝 결과 ──────────────────────────────────────────────────────────
with tab4:
    st.subheader(":material/tune: 하이퍼파라미터 그리드 탐색 결과")

    fig4 = go.Figure(go.Bar(
        x=tuning_df.apply(
            lambda r: f"cps={r['changepoint_prior_scale']} / sps={r['seasonality_prior_scale']} / {r['seasonality_mode']}",
            axis=1,
        ),
        y=tuning_df["MAPE(%)"],
        marker_color=["#c43d5c" if i == 0 else "#cccccc" for i in range(len(tuning_df))],
        text=tuning_df["MAPE(%)"].round(2), textposition="outside",
    ))
    fig4.update_layout(
        xaxis_title="파라미터 조합", yaxis_title="MAPE (%)",
        height=400, xaxis_tickangle=-20, margin=dict(b=120),
    )
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown("**전체 탐색 결과**")
    st.dataframe(tuning_df, use_container_width=True, hide_index=True)

    best_row = tuning_df.iloc[0]
    st.success(
        f"최적 설정 — "
        f"changepoint_prior_scale: **{best_row['changepoint_prior_scale']}** / "
        f"seasonality_prior_scale: **{best_row['seasonality_prior_scale']}** / "
        f"mode: **{best_row['seasonality_mode']}** "
        f"→ MAPE **{best_row['MAPE(%)']:.2f}%**"
    )

    tuning_text = tuning_df.to_string(index=False)
    ai_explain_button("explain_tuning", f"""
Prophet 모델의 하이퍼파라미터 그리드 탐색 결과야.

{tuning_text}

파라미터 설명:
- changepoint_prior_scale (cps): 트렌드 변화 민감도. 높을수록 트렌드가 유연하게 꺾임
- seasonality_prior_scale (sps): 계절성 강도. 높을수록 계절 패턴을 강하게 반영
- seasonality_mode: additive(계절 효과가 고정 크기) vs multiplicative(트렌드에 비례)

어떤 파라미터 조합이 왜 좋은 성능을 냈는지, 그리고 이 데이터의 특성상 왜 그 조합이 적합한지 3~4문장으로 설명해줘.
""")
