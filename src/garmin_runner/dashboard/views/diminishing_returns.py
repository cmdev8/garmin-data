from __future__ import annotations

from garmin_runner.dashboard.visualizations import (
    diminishing_returns_history_chart,
    diminishing_returns_history_context_chart,
    diminishing_returns_history_table,
    diminishing_returns_values,
)


def render(data, config=None) -> None:
    import streamlit as st

    st.subheader("Diminishing returns")
    st.caption(
        "Ez az oldal óvatos becslést mutat arról, hogy ugyanaz az edzésinger várhatóan kisebb improvementet hoz-e magasabb edzettségnél."
    )

    values = diminishing_returns_values(data.ml_result)
    cols = st.columns(3)
    for index, (label, value) in enumerate(values.items()):
        cols[index % 3].metric(label, value)

    metrics = data.ml_result.get("metrics", {})
    reason = str(metrics.get("diminishing_returns_reason", "n/a"))
    if reason == "learned_from_history":
        st.success("A diminishing returns factor a feltöltött előzményekből tanult csökkenő marginális választ használja, de a hatás konzervatívan korlátozott.")
    elif reason in {"insufficient_true_7d_history", "too_little_fitness_variation"}:
        st.info("Nincs elég változatos történeti adat, ezért a diminishing returns factor semleges: 100%.")
    elif reason == "non_diminishing_history":
        st.info("A feltöltött adatok nem mutattak diminishing returnst, ezért a pozitív tervhatás nincs csökkentve.")
    else:
        st.info("A diminishing returns factor semleges vagy fallback állapotból származik.")

    st.subheader("Havi diminishing returns factor")
    st.caption(
        "A havi sor minden pontja csak az adott hónap végéig ismert, lezárt 7 napos válaszmegfigyeléseket használja."
    )
    history_chart = diminishing_returns_history_chart(data.ml_result)
    context_chart = diminishing_returns_history_context_chart(data.ml_result)
    history_table = diminishing_returns_history_table(data.ml_result)
    if history_chart.empty:
        st.info("Nincs elég havi történeti adat a hozamfaktor idősorához.")
    else:
        st.line_chart(history_chart.set_index("Hónap")["diminishing returns factor (%)"])
        st.caption(
            "A magasabb factor nem jobb teljesítményt jelent, hanem kisebb levonást az optimista tervhatásból. "
            "A factor emelkedhet, ha az aktuális performance index alacsonyabb, vagy a hónap végéig ismert előzmények alapján "
            "a model nagyobb relatív választ becsül ugyanarra a training loadra."
        )
        if not context_chart.empty:
            st.line_chart(context_chart.set_index("Hónap"))
        st.dataframe(history_table, width="stretch", hide_index=True)

    st.markdown(
        """
**Mit jelent?**

- A modell a 7 napos edzésterhelés utáni teljesítményindex-változást vizsgálja.
- Ha magasabb teljesítményszinten kisebb improvement látszik ugyanannyi training loadra, a pozitív tervhatás becslése csökken.
- Ha a factor később emelkedik, az nem azt jelenti, hogy a diminishing returns „javul”, hanem hogy a pont-in-time becslés szerint épp kevesebb korrekció kell.
- A korrekció legfeljebb mérsékelt: a lineáris illesztés nem veheti le irreálisan alacsonyra a terv várható hasznát.
- A faktor csak az optimista javulást csökkenti; a fáradtság vagy túlterhelés miatti negatív hatást nem enyhíti.
"""
    )
