from __future__ import annotations


ALGORITHM_EQUATIONS = [
    "load = distance_km * 10 + hard_time_min * 1.5 + moderate_time_min",
    "stimulus_t = load_t + 0.8 * hard_time_t + 0.3 * moderate_time_t + min(25, 1.5 * distance_t)",
    "fitness_t = 0.965 * fitness_{t-1} + 0.035 * stimulus_t",
    "fatigue_t = 0.82 * fatigue_{t-1} + 0.18 * stimulus_t",
    "form_t = fitness_t - fatigue_t",
    "performance_index_t = max(1, fitness_t + 0.25 * form_t)",
    "overload_risk_t = clamp((fatigue_t - fitness_t) / 50, 0, 1)",
    "T2 = T1 * (D2 / D1)^1.06",
    "pace2 = T2 / D2",
    "durability = max(1, 1.12 - min(0.12, long_run_km / (distance_km * 10)))",
    "frequency_factor = min(1, comparable_distance_count / required_count)",
    "distance_factor = min(1, longest_run_km / target_distance_km)",
    "evidence_factor = min(frequency_factor, distance_factor)",
    "scarcity_multiplier = 1 + max_penalty * (1 - evidence_factor)",
    "target_t = performance_index_{t+7}",
    "x_t = features_{<=t-window:t}",
    "temporal_feature = {latest, lag, mean, min, max, slope, delta, volatility}",
    "TimeSeriesSplit: train_index < validation_index",
    "confidence = clamp(0.25 + 0.45 * min(0.9, n/45) + 0.3 * max(0, 1 - MAE/25), 0.1, 0.9)",
    "response_i = max(0, performance_index_{i+7} - performance_index_i) / max(0.25, load_7d_i / 100)",
    "response_i = alpha + beta * performance_index_i",
    "raw_ratio = (alpha + beta * latest_performance_index) / (alpha + beta * p25(performance_index))",
    "diminishing_returns_factor = clamp(1 - 0.35 * (1 - clamp(raw_ratio, 0, 1)), 0.65, 1.0)",
    "adjusted_improvement = improvement * diminishing_returns_factor if improvement > 0 else improvement",
    "historical_driver_priority = improvement_s_per_km * confidence",
    "historical_driver_match = clamp(driver_score, 0, 1) if candidate_family_matches_driver else 0",
    "score = 2.0 * expected_adaptation + 1.5 * missing_stimulus_score + 1.0 * race_specificity + 0.8 * historical_driver_match - 2.0 * expected_fatigue - 3.0 * overload_risk - 2.0 * constraint_penalty",
    "expected_adaptation = 0.65 * deterministic_adaptation + 0.35 * ml_expected_adaptation",
    "overload_risk = 0.75 * deterministic_overload + 0.25 * ml_overload_modifier",
    "planned_discount = clamp(0.30 * walk_fraction + 0.10 * planned_run_walk_score, 0.0, 0.25)",
    "run_walk_fatigue_multiplier = clamp(1.0 - planned_discount, 0.75, 1.0)",
    "fatigue_adjusted_load = load * run_walk_fatigue_multiplier",
    "fatigue_t = 0.82 * fatigue_{t-1} + 0.18 * fatigue_adjusted_stimulus_t",
]

LATEX_EQUATIONS = [
    r"\mathrm{load} = \mathrm{distance}_{km} \cdot 10 + \mathrm{hard\_time}_{min} \cdot 1.5 + \mathrm{moderate\_time}_{min}",
    r"\mathrm{stimulus}_t = \mathrm{load}_t + 0.8 \cdot \mathrm{hard\_time}_t + 0.3 \cdot \mathrm{moderate\_time}_t + \min(25, 1.5 \cdot \mathrm{distance}_t)",
    r"\mathrm{fitness}_t = 0.965 \cdot \mathrm{fitness}_{t-1} + 0.035 \cdot \mathrm{stimulus}_t",
    r"\mathrm{fatigue}_t = 0.82 \cdot \mathrm{fatigue}_{t-1} + 0.18 \cdot \mathrm{stimulus}_t",
    r"\mathrm{form}_t = \mathrm{fitness}_t - \mathrm{fatigue}_t",
    r"\mathrm{performance\_index}_t = \max(1, \mathrm{fitness}_t + 0.25 \cdot \mathrm{form}_t)",
    r"\mathrm{overload\_risk}_t = \mathrm{clamp}\left(\frac{\mathrm{fatigue}_t - \mathrm{fitness}_t}{50}, 0, 1\right)",
    r"T_2 = T_1 \cdot \left(\frac{D_2}{D_1}\right)^{1.06}",
    r"\mathrm{pace}_2 = \frac{T_2}{D_2}",
    r"\mathrm{durability} = \max\left(1, 1.12 - \min\left(0.12, \frac{\mathrm{long\_run}_{km}}{\mathrm{distance}_{km} \cdot 10}\right)\right)",
    r"\mathrm{frequency\_factor} = \min\left(1, \frac{\mathrm{comparable\_distance\_count}}{\mathrm{required\_count}}\right)",
    r"\mathrm{distance\_factor} = \min\left(1, \frac{\mathrm{longest\_run}_{km}}{\mathrm{target\_distance}_{km}}\right)",
    r"\mathrm{evidence\_factor} = \min(\mathrm{frequency\_factor}, \mathrm{distance\_factor})",
    r"\mathrm{scarcity\_multiplier} = 1 + \mathrm{max\_penalty} \cdot (1 - \mathrm{evidence\_factor})",
    r"\mathrm{target}_t = \mathrm{performance\_index}_{t+7}",
    r"x_t = \mathrm{features}_{\le t-\mathrm{window}:t}",
    r"\mathrm{temporal\_feature} \in \{\mathrm{latest}, \mathrm{lag}, \mathrm{mean}, \mathrm{min}, \mathrm{max}, \mathrm{slope}, \mathrm{delta}, \mathrm{volatility}\}",
    r"\mathrm{TimeSeriesSplit}: \max(\mathrm{train\_index}) < \min(\mathrm{validation\_index})",
    r"\mathrm{confidence} = \mathrm{clamp}\left(0.25 + 0.45 \cdot \min(0.9, n/45) + 0.3 \cdot \max(0, 1 - \mathrm{MAE}/25), 0.1, 0.9\right)",
    r"\mathrm{response}_i = \frac{\max(0, \mathrm{performance\_index}_{i+7} - \mathrm{performance\_index}_i)}{\max(0.25, \mathrm{load\_7d}_i / 100)}",
    r"\mathrm{response}_i = \alpha + \beta \cdot \mathrm{performance\_index}_i",
    r"\mathrm{raw\_ratio} = \frac{\alpha + \beta \cdot \mathrm{latest\_performance\_index}}{\alpha + \beta \cdot p25(\mathrm{performance\_index})}",
    r"\mathrm{diminishing\_returns\_factor} = \mathrm{clamp}(1 - 0.35 \cdot (1 - \mathrm{clamp}(\mathrm{raw\_ratio}, 0, 1)), 0.65, 1.0)",
    r"\mathrm{adjusted\_improvement} = \begin{cases}\mathrm{improvement} \cdot \mathrm{diminishing\_returns\_factor}, & \mathrm{improvement} > 0\\\mathrm{improvement}, & \mathrm{improvement} \le 0\end{cases}",
    r"\mathrm{historical\_driver\_priority} = \mathrm{improvement}_{s/km} \cdot \mathrm{confidence}",
    r"\mathrm{historical\_driver\_match} = \begin{cases}\mathrm{clamp}(\mathrm{driver\_score}, 0, 1), & \mathrm{candidate\_family\_matches\_driver}\\0, & \mathrm{otherwise}\end{cases}",
    r"\mathrm{score} = 2.0 \cdot \mathrm{expected\_adaptation} + 1.5 \cdot \mathrm{missing\_stimulus\_score} + 1.0 \cdot \mathrm{race\_specificity} + 0.8 \cdot \mathrm{historical\_driver\_match} - 2.0 \cdot \mathrm{expected\_fatigue} - 3.0 \cdot \mathrm{overload\_risk} - 2.0 \cdot \mathrm{constraint\_penalty}",
    r"\mathrm{expected\_adaptation} = 0.65 \cdot \mathrm{deterministic\_adaptation} + 0.35 \cdot \mathrm{ml\_expected\_adaptation}",
    r"\mathrm{overload\_risk} = 0.75 \cdot \mathrm{deterministic\_overload} + 0.25 \cdot \mathrm{ml\_overload\_modifier}",
    r"\mathrm{planned\_discount} = \mathrm{clamp}(0.30 \cdot \mathrm{walk\_fraction} + 0.10 \cdot \mathrm{planned\_run\_walk\_score}, 0.0, 0.25)",
    r"\mathrm{run\_walk\_fatigue\_multiplier} = \mathrm{clamp}(1.0 - \mathrm{planned\_discount}, 0.75, 1.0)",
    r"\mathrm{fatigue\_adjusted\_load} = \mathrm{load} \cdot \mathrm{run\_walk\_fatigue\_multiplier}",
    r"\mathrm{fatigue}_t = 0.82 \cdot \mathrm{fatigue}_{t-1} + 0.18 \cdot \mathrm{fatigue\_adjusted\_stimulus}_t",
]


def render(data=None, config=None) -> None:
    import streamlit as st

    st.header("Dokumentáció")
    st.markdown(
        """
Ez az oldal azt foglalja össze, hogy az alkalmazás milyen szabályokat,
képleteket és machine learning lépéseket használ. A leírás a jelenlegi
implementációt követi, nem általános edzéselméleti ajánlás.
"""
    )

    st.subheader("Adattisztítás és szűrés")
    st.markdown(
        """
- A FIT fájloknál először a `session.sport` és `session.sub_sport` mezők alapján történik a szűrés.
- A futás jellegű aktivitások maradnak meg; a kerékpár, úszás, erősítés és hasonló sportok kiesnek.
- Ha a sportmetaadat hiányzik, a rendszer távolság, idő, sebesség, lépésütem és GPS-jelek alapján dönt.
- Futás-séta edzéseknél a teljes átlagtempó csak logisztikai adat; a teljesítményjelhez futás közbeni tempót használunk, ha elérhető.
"""
    )

    st.subheader("FIT-ből kinyert feature-ök")
    st.markdown(
        """
A model nem csak heti km-et és időt lát. Ahol a feltöltött Garmin FIT fájlok
tartalmazzák, opcionális feature-ként bekerülnek:

- terhelési és energiamezők: kalória, aerob és anaerob `training effect`, munka;
- pulzusmezők: minimum/átlag/maximum pulzus, pulzusdrift, pulzusvariabilitás;
- teljesítménymérő mezők: átlagteljesítmény, maximum teljesítmény, normalizált teljesítmény, teljesítmény/pulzus hatékonyság;
- futódinamika: lépésütem, lépéshossz, vertikális oszcilláció, vertikális arány, talajkontakt idő;
- környezet és útvonal-proxy: hőmérséklet, szintemelkedés kilométerenként, magasságtartomány, GPS lefedettség;
- kivitelezési minőség: tempó-, kör-, pulzus-, lépésütem- és teljesítményvariabilitás;
- HRV összegzés: mintaszám, medián RR és RMSSD, alacsony bizalmú kísérleti jelként.

Az eszközazonosító, sorozatszám és nyers GPS koordináta nem prediktív feature.
A terep továbbra is csak szintemelkedés/útvonal proxy, mert a FIT fájl nem ad
megbízható felszíntípust.
"""
    )

    st.subheader("Napi terhelés")
    st.latex(LATEX_EQUATIONS[0])
    st.markdown("A napi training load egyszerű determinisztikus becslés: a táv az alap, a minőségi/kemény és közepes intenzitás többletsúlyt kap.")

    st.subheader("Futás-séta fáradtsági korrekció")
    for equation in LATEX_EQUATIONS[29:]:
        st.latex(equation)
    st.markdown(
        """
Tervezett futás-séta edzéseknél a sétaszakaszok csökkentik a folyamatos
mechanikai és metabolikus training loadot, ezért a fatigue load alacsonyabb,
mint az azonos távolságú folyamatos futásé. Kényszerű sétaszüneteknél nincs
ilyen kedvezmény, mert ezek inkább felhalmozott fáradtságot vagy rossz
tempóválasztást jeleznek.
"""
    )

    st.subheader("Fitness-fatigue modell")
    for equation in LATEX_EQUATIONS[1:7]:
        st.latex(equation)
    st.markdown(
        """
A fitness lassabban csökkenő állapotváltozó, a fatigue gyorsabban reagál.
A `form` a kettő különbsége, a performance index pedig konzervatív becslés
az aktuális állapotra.
"""
    )

    st.subheader("Versenybecslés")
    for equation in LATEX_EQUATIONS[7:14]:
        st.latex(equation)
    st.markdown(
        """
A baseline tempó a futás közbeni tempóból, a legjobb gördülő 5 km-ből vagy végső
fallbackként az átlagtempóból származik. A hosszabb távoknál a tartóssági
büntetés mellett külön `scarcity_multiplier` is van. Félmaratonnál a közvetlen
evidence a `18–25 km` közötti futás; `required_count = 3`, a maximális scarcity
penalty `12%`. Maratonnál a közvetlen evidence az `>=30 km` futás, a `18–30 km`
tartomány csak support evidence; `required_count = 3`, a maximális scarcity
penalty `25%`. A prediction nem lehet gyorsabb a kevés, de releváns hosszú futás
bizonyítéknál, ha nincs elég hasonló távú előzmény.
"""
    )

    st.subheader("Pulzus- és tempózónák")
    st.markdown(
        """
Pulzuszónák a laktátküszöb pulzusához képest:

- Z1: `resting_hr` – `0.80 * LTHR`
- Z2: `0.80 * LTHR` – `0.88 * LTHR`
- Z3: `0.88 * LTHR` – `0.95 * LTHR`
- Z4: `0.95 * LTHR` – `1.02 * LTHR`
- Z5: `1.02 * LTHR` – `max_hr`

A tempózónák a küszöbtempó szorzóiból készülnek: regeneráló, laza,
egyenletes, maraton, küszöb, VO2 és résztáv tartományok.
"""
    )

    st.subheader("Machine learning")
    st.latex(LATEX_EQUATIONS[14])
    st.latex(LATEX_EQUATIONS[15])
    st.latex(LATEX_EQUATIONS[16])
    st.latex(LATEX_EQUATIONS[17])
    st.latex(LATEX_EQUATIONS[18])
    for equation in LATEX_EQUATIONS[19:24]:
        st.latex(equation)
    st.markdown(
        """
Minden elemzéskor friss, memóriában élő model tanul. Elég adat esetén
sklearn-native temporal model fut: a heti sorokból 7/14/28 napos lookback
ablakok készülnek, majd `TimeSeriesSplit` validáció ellenőrzi, hogy korábbi
időszakból tanulunk és későbbi időszakon mérünk. A fő model `RidgeCV`, mert a
koefficiensekből pozitív és negatív temporal driverek magyarázhatók.
Legalább 18 használható történeti sor kell a temporal úthoz; nagyobb történetnél
`HistGradientBoostingRegressor` is lehet jelölt ugyanazon temporal feature-ökön.
A nonlinear jelölt csak akkor nyerhet, ha a kronologikus MAE egyértelműen jobb;
közeli MAE esetén a `RidgeCV` marad a magyarázhatóbb választás. Kevés adatnál
vagy konstans célváltozónál `DummyRegressor` ad biztonságos fallback predictiont.
A model nem kerül fájlba, csak a metric-ek, predictionök és magyarázó adatok
maradnak a dashboard memóriájában.

A tanító feature-ök napi/heti aggregátumokból készülnek, csak az adott napig
ismert adatokból: táv, idő, raw és fatigue-adjusted training load, intenzitás,
pulzus, power, futódinamika, hőmérséklet, szintemelkedés, HRV és variability
mutatók. A hiányzó FIT-mezők nem okoznak hibát; a model nullával imputált
oszlopként kezeli őket, ezért a feature importance csak akkor kap értelmet,
ha az adott mező ténylegesen jelen volt az előzményekben.

A `HistGradientBoostingRegressor` hisztogram-alapú döntési fákat illeszt:
a folytonos jellemzőket gyors `histogram binning` lépés diszkrét vödrökbe
rendezi, majd a modell additív `gradient boosting` módon egymás után tanuló
fákkal csökkenti a négyzetes hibát. A jelenlegi beállítás `random_state=42`
és `max_iter=80`. Temporal jelöltként `TimeSeriesSplit` validációban vesz részt.
Ha a temporal út nem használható, tabular fallbackként az időrend korábbi részén
tanul és későbbi sorokon validál; ez kronologikus holdout, nem random split és
nem a fő temporal út. A validációs hiba `MAE`, a feature magyarázatát pedig
`permutation importance` adja, ha van legalább négy validációs sor.

A temporal explanation a lag/window feature-öket visszacsoportosítja olvasható
base feature nevekre. Lineáris temporal modelnél a pozitív koefficiens jobb
jövőbeli performance indexhez, a negatív koefficiens rosszabb jövőbeli
performance indexhez kapcsolódó jelként jelenik meg. Nemlineáris modelnél
előjeles magyarázat helyett `permutation importance` látható.

A diminishing returnst az alkalmazás a saját edzéstörténetből becsüli: azt nézi,
hogy azonosított 7 napos training load után mekkora performance index improvement
jelent meg alacsonyabb és magasabb edzettségnél. Ha a történet szerint magasabb
edzettségnél kisebb a marginális válasz, a `diminishing returns factor` csak a
pozitív tervhatást csökkenti; a fatigue vagy overload risk miatti negatív hatást
nem enyhíti.
"""
    )

    st.subheader("V2 optimizer")
    for equation in LATEX_EQUATIONS[24:29]:
        st.latex(equation)
    st.markdown(
        """
Az optimizer több jelölt edzéshetet készít, kizárja a veszélyeseket
hard constraint szabályokkal, majd score-olja a maradékot. A machine learning akkor keveredik
be a scoringba, ha a model confidence legalább `0.35`; különben a determinisztikus
score dominál. Futás-séta előírás csak akkor kerül jelölt tervbe, ha a futás-séta
támogatás és a futás-séta jelölttervezés is explicit engedélyezett.

A kontrollált Teljesítménytrend oldalból az optimizer kinyeri a legerősebb
akcióképes történeti improvement drivert. Például ha a korábbi javulás
leginkább küszöbedzéshez, VO2max-hoz vagy futótechnikához kapcsolódott, akkor
safe állapotban több quality vagy speed-support jelölt kaphat plusz score-t.
Ez csak soft bias: magas fatigue, magas overload risk, rossz form vagy forced
run-walk esetén a hard-run jel elnyomódik, és a biztonsági szabályok maradnak
az erősebbek.
"""
    )
