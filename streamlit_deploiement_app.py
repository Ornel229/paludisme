import streamlit as st
import streamlit.components.v1 as components
import base64
import unicodedata
import joblib
import pandas as pd


# ============================================================
# MOTEUR DE PRÉDICTION (repris de fast_api.py, exécuté ici
# directement dans Streamlit — plus besoin d'API séparée)
# ============================================================

NOMS_MOIS = [
    "",
    "Janvier", "Février", "Mars", "Avril",
    "Mai", "Juin", "Juillet", "Août",
    "Septembre", "Octobre", "Novembre", "Décembre",
]

FEATURES_MODELE = [
    "commune", "annee", "mois",
    "temperature_moy_C", "temperature_max_C",
    "precipitation_mm", "couverture_moustiquaires", "acces_assainissement_pct",
]


def normaliser(texte: str) -> str:
    """Normalise un nom de commune pour la comparaison (accents, tirets, casse)."""
    texte = texte.strip().lower()
    texte = unicodedata.normalize("NFD", texte)
    texte = "".join(c for c in texte if unicodedata.category(c) != "Mn")
    texte = texte.replace("-", " ").replace("'", " ").replace("\u2019", " ")
    texte = " ".join(texte.split())
    return texte


@st.cache_resource
def charger_modele_et_reference():
    modele = joblib.load("modele_paludisme.pkl")
    ref_communes = pd.read_csv("reference_communes.csv")
    ref_communes["commune_norm"] = ref_communes["commune"].apply(normaliser)
    return modele, ref_communes


def calculer_estimation(commune, annee, mois, temperature_moy_C, temperature_max_C,
                         precipitation_mm, couverture_moustiquaires, acces_assainissement_pct):
    modele, ref_communes = charger_modele_et_reference()

    if temperature_max_C < temperature_moy_C:
        raise ValueError(
            "La température maximale ne peut pas être inférieure à la température moyenne."
        )

    commune_norm = normaliser(commune)
    ref = ref_communes[ref_communes["commune_norm"] == commune_norm]

    if ref.empty:
        communes_dispo = ", ".join(sorted(ref_communes["commune"].tolist()))
        raise ValueError(
            f"Commune non reconnue : '{commune}'. Communes disponibles : {communes_dispo}"
        )

    commune_officielle = ref["commune"].iloc[0]
    departement        = ref["departement"].iloc[0]
    pop_2002           = ref["Population_2002"].iloc[0]
    taux_croissance    = ref["taux_decimal"].iloc[0]

    ligne = pd.DataFrame([{
        "commune":                  commune_officielle,
        "annee":                    annee,
        "mois":                     mois,
        "temperature_moy_C":        temperature_moy_C,
        "temperature_max_C":        temperature_max_C,
        "precipitation_mm":         precipitation_mm,
        "couverture_moustiquaires": couverture_moustiquaires,
        "acces_assainissement_pct": acces_assainissement_pct,
    }])
    ligne["commune"] = pd.Categorical(
        ligne["commune"], categories=ref_communes["commune"].unique()
    )
    ligne = ligne[FEATURES_MODELE]

    pred_brute         = modele.predict(ligne)[0]
    population_estimee = pop_2002 * ((1 + taux_croissance) ** (annee - 2002))

    cas_valides = max(0, round(pred_brute))
    cas_final   = min(cas_valides, int(population_estimee))

    return {
        "cas_estimes":        cas_final,
        "population_estimee": int(population_estimee),
        "departement":        departement,
        "commune":            commune_officielle,
        "mois":               NOMS_MOIS[mois],
        "annee":              annee,
    }


@st.cache_data
def liste_communes_disponibles():
    _, ref_communes = charger_modele_et_reference()
    return sorted(ref_communes["commune"].tolist())


# ============================================================
# CONFIGURATION DE LA PAGE
# ============================================================

st.set_page_config(
    page_title="Prédiction du Paludisme",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# IMAGE DE FOND + CSS
# ============================================================

def mettre_image_fond(image_path):

    with open(image_path, "rb") as image:
        image_base64 = base64.b64encode(image.read()).decode()

    css = f"""
    <style>

    /* Fond */
    .stApp {{
        background-image:
            linear-gradient(rgba(0,0,0,0.72), rgba(0,0,0,0.78)),
            url("data:image/png;base64,{image_base64}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}

    .main .block-container {{
        max-width: 1200px;
        padding-top: 40px;
        padding-bottom: 50px;
    }}

    /* Blanc global */
    .stApp p,
    .stApp span,
    .stApp li,
    .stApp h1,
    .stApp h2,
    .stApp h3,
    .stApp h4,
    .stMarkdown,
    .stMarkdown p,
    .stMarkdown span,
    [data-testid="stMarkdownContainer"],
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] span,
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3 {{
        color: #ffffff !important;
        font-family: Arial, sans-serif !important;
    }}

    /* Labels des widgets */
    [data-testid="stWidgetLabel"],
    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] span,
    [data-testid="stWidgetLabel"] label {{
        color: #ffffff !important;
        font-size: 15px !important;
        font-weight: 600 !important;
    }}

    /* Titre */
    .titre {{
        text-align: center;
        color: #ffffff !important;
        font-size: 45px;
        font-weight: 700;
        margin-bottom: 5px;
        font-family: Arial, sans-serif;
    }}
    .titre span {{
        color: #ff3045 !important;
    }}
    .sous-titre {{
        text-align: center;
        color: #dddddd !important;
        font-size: 18px;
        margin-bottom: 35px;
        font-family: Arial, sans-serif;
    }}

    /* Champ texte */
    .stTextInput > div > div > input {{
        background-color: #1e2a38 !important;
        color: #ffffff !important;
        border: 1px solid rgba(255,255,255,0.35) !important;
        border-radius: 6px !important;
        font-size: 15px !important;
        caret-color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }}
    .stTextInput > div > div > input::placeholder {{
        color: rgba(200,200,200,0.5) !important;
        -webkit-text-fill-color: rgba(200,200,200,0.5) !important;
    }}
    .stTextInput > div > div > input:focus {{
        border-color: #ff3045 !important;
        box-shadow: 0 0 0 1px #ff3045 !important;
    }}

    /* Champ numérique */
    .stNumberInput > div > div > input {{
        background-color: #1e2a38 !important;
        color: #ffffff !important;
        border: 1px solid rgba(255,255,255,0.35) !important;
        border-radius: 6px !important;
        font-size: 15px !important;
        caret-color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }}
    .stNumberInput > div > div > input:focus {{
        border-color: #ff3045 !important;
        box-shadow: 0 0 0 1px #ff3045 !important;
    }}
    .stNumberInput button {{
        background-color: #253444 !important;
        color: #ffffff !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
    }}
    .stNumberInput button p {{
        color: #ffffff !important;
    }}

    /* Selectbox — boîte principale */
    .stSelectbox > div > div {{
        background-color: #1e2a38 !important;
        border: 1px solid rgba(255,255,255,0.35) !important;
        border-radius: 6px !important;
    }}
    .stSelectbox > div > div > div > div {{
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        font-size: 15px !important;
        background-color: #1e2a38 !important;
    }}
    .stSelectbox svg {{
        fill: #ffffff !important;
    }}

    /* Selectbox — menu déroulant
       Le menu est rendu HORS du stApp, dans un portail du DOM.
       On cible donc le body directement. */
    body [data-baseweb="popover"],
    body [data-baseweb="popover"] *,
    body [data-baseweb="menu"],
    body [data-baseweb="menu"] * {{
        background-color: #1a2535 !important;
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        font-family: Arial, sans-serif !important;
    }}
    body [data-baseweb="menu"] [role="option"]:hover,
    body [data-baseweb="menu"] li:hover {{
        background-color: rgba(255,48,69,0.3) !important;
    }}
    body [data-baseweb="menu"] [aria-selected="true"] {{
        background-color: rgba(255,48,69,0.45) !important;
    }}

    /* Bouton Prédire */
    .stButton > button {{
        width: 100%;
        height: 52px;
        background-color: #ff3045 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        font-size: 17px !important;
        font-weight: 600 !important;
    }}
    .stButton > button:hover {{
        background-color: #e52338 !important;
    }}
    .stButton > button p {{
        color: #ffffff !important;
    }}

    /* Résultat */
    .resultat {{
        background: rgba(12,20,29,0.96);
        border: 1px solid rgba(255,48,69,0.45);
        border-radius: 15px;
        padding: 30px;
        text-align: center;
        margin-top: 25px;
    }}
    .resultat-titre {{
        color: #cccccc !important;
        font-size: 17px;
        font-family: Arial, sans-serif;
    }}
    .nombre {{
        color: #ff3045 !important;
        font-size: 52px;
        font-weight: bold;
        margin: 8px 0;
        font-family: Arial, sans-serif;
    }}
    .details {{
        color: #dddddd !important;
        font-size: 15px;
        font-family: Arial, sans-serif;
    }}

    /* Footer */
    .footer {{
        text-align: center;
        color: #aaaaaa !important;
        font-size: 13px;
        margin-top: 35px;
        font-family: Arial, sans-serif;
    }}

    </style>
    """

    st.markdown(css, unsafe_allow_html=True)


mettre_image_fond("paludisme_image.png")


# ============================================================
# TITRE
# ============================================================

st.markdown(
    """
    <div class="titre">
        Prédiction du <span>Paludisme</span>
    </div>
    <div class="sous-titre">
        Estimation du nombre de cas de paludisme par commune au Bénin
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CARTES À PROPOS + OBJECTIF
# Rendu via components.html() — jamais échappé par Streamlit
# ============================================================

components.html(
    """
    <style>
        body { margin: 0; padding: 0; background: transparent; }
        .conteneur {
            display: flex;
            justify-content: center;
            gap: 30px;
            flex-wrap: wrap;
            padding: 0 10px 10px 10px;
        }
        .carte {
            background: rgba(10, 16, 24, 0.92);
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 15px;
            padding: 28px 32px;
            flex: 1;
            min-width: 260px;
            max-width: 520px;
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.35);
            font-family: Arial, sans-serif;
        }
        .carte h3 {
            color: #ffffff;
            font-size: 20px;
            margin: 0 0 14px 0;
        }
        .carte p {
            color: #cccccc;
            font-size: 15px;
            line-height: 1.75;
            margin: 0;
        }
    </style>

    <div class="conteneur">

        <div class="carte">
            <h3>À propos</h3>
            <p>
                Cette application estime le nombre de cas de paludisme
                à partir de facteurs environnementaux et socio-démographiques :
                commune, période, températures, précipitations,
                couverture en moustiquaires et accès à l'assainissement.
            </p>
        </div>

        <div class="carte">
            <h3>Objectif</h3>
            <p>
                Fournir une estimation fiable pour mieux analyser
                l'évolution du paludisme au Bénin et appuyer
                les actions de prévention sur le terrain.
            </p>
        </div>

    </div>
    """,
    height=180,
    scrolling=False
)


# ============================================================
# FORMULAIRE — pleine largeur
# ============================================================

liste_communes = liste_communes_disponibles()

col_a, col_b = st.columns(2)

with col_a:
    if liste_communes:
        commune = st.selectbox("Commune", liste_communes)
    else:
        commune = st.text_input(
            "Commune",
            placeholder="Liste des communes indisponible — saisissez manuellement"
        )
    temperature_moy = st.number_input(
        "Température moyenne (°C)",
        min_value=15.0, max_value=45.0, value=28.0, step=0.1
    )
    precipitation = st.number_input(
        "Précipitations (mm)", min_value=0.0, value=100.0, step=1.0
    )
    moustiquaires = st.number_input(
        "Couverture en moustiquaires (%)",
        min_value=0.0, max_value=100.0, value=50.0, step=0.1
    )

with col_b:
    mois_noms = [
        "Janvier", "Février", "Mars", "Avril",
        "Mai", "Juin", "Juillet", "Août",
        "Septembre", "Octobre", "Novembre", "Décembre"
    ]
    mois_nom = st.selectbox("Mois", mois_noms)
    mois = mois_noms.index(mois_nom) + 1
    annee = st.number_input(
        "Année", min_value=2002, max_value=2100, value=2026, step=1
    )
    temperature_max = st.number_input(
        "Température maximale (°C)",
        min_value=15.0, max_value=55.0, value=32.0, step=0.1
    )
    assainissement = st.number_input(
        "Accès à l'assainissement (%)",
        min_value=0.0, max_value=100.0, value=70.0, step=0.1
    )


st.write("")
lancer_prediction = st.button("Prédire")


# ============================================================
# PRÉDICTION
# ============================================================

if lancer_prediction:

    if not commune or not commune.strip():
        st.error("Veuillez sélectionner une commune.")

    elif temperature_max < temperature_moy:
        st.error(
            "La température maximale ne peut pas être "
            "inférieure à la température moyenne."
        )

    else:

        try:
            resultat = calculer_estimation(
                commune=commune.strip(),
                annee=int(annee),
                mois=int(mois),
                temperature_moy_C=float(temperature_moy),
                temperature_max_C=float(temperature_max),
                precipitation_mm=float(precipitation),
                couverture_moustiquaires=float(moustiquaires),
                acces_assainissement_pct=float(assainissement),
            )

            cas         = resultat["cas_estimes"]
            population  = resultat["population_estimee"]
            departement = resultat["departement"]
            nom_mois    = resultat["mois"]

            st.markdown(
                f"""
                <div class="resultat">
                    <div class="resultat-titre">Nombre de cas estimés</div>
                    <div class="nombre">{cas:,}</div>
                    <div class="details">
                        {commune} — {departement}<br>
                        {nom_mois} {annee}<br><br>
                        Population estimée : {population:,} habitants
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        except ValueError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Une erreur est survenue : {e}")


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        Prédiction du paludisme au Bénin
    </div>
    """,
    unsafe_allow_html=True
)

# streamlit run streamlit_app.py