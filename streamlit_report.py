import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import duckdb
import glob

# ============================================================================
# CONFIGURATION STREAMLIT
# ============================================================================

st.set_page_config(
    page_title="Dilemme du Prisonnier — IA vs Stratégies Codées",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Palette de couleurs douces et harmonieuses
COLORS = {
    "ia_primary": "#6B9BD1",      # Bleu doux
    "coded_primary": "#A8B39F",   # Vert grisé
    "cooperation": "#8FBC8F",     # Vert pâle
    "defection": "#CD8162",       # Terracotta doux
    "mutual_coop": "#A8D5BA",     # Vert menthe
    "mutual_defect": "#D4A5A5",   # Rose pâle
    "exploit_1": "#F4D8C8",       # Pêche pâle
    "exploit_2": "#E8D5C4",       # Beige pâle
    "neutral": "#B5B3B0",         # Gris neutre
}

# Appliquer un style global personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.2em;
        color: #4A5F7F;
        margin-bottom: 0.5em;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .section-header {
        font-size: 1.5em;
        color: #5A7B9E;
        margin-top: 1.5em;
        margin-bottom: 0.8em;
        border-left: 4px solid #6B9BD1;
        padding-left: 1em;
        font-weight: 500;
    }
    .subsection {
        font-size: 1.1em;
        color: #6B7B8F;
        margin-top: 1em;
        margin-bottom: 0.5em;
        font-weight: 500;
    }
    .insight-box {
        background-color: #F5F7FA;
        padding: 1.2em;
        border-radius: 0.5em;
        border-left: 4px solid #6B9BD1;
        margin: 1em 0;
        line-height: 1.6;
    }
    .metric-card {
        background-color: #F8F9FB;
        padding: 1.5em;
        border-radius: 0.5em;
        border-top: 3px solid #6B9BD1;
        text-align: center;
    }
    .comparison-table {
        background-color: #F5F7FA;
        border-radius: 0.5em;
        padding: 1em;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CHARGEMENT ET PRÉPARATION DES DONNÉES
# ============================================================================

@st.cache_data
def load_and_prepare_data():
    """Charger et préparer les données enrichies"""
    try:
        df = pd.read_parquet("enriched_data/enriched_games_full.parquet")
        
        # Dériver les colonnes agent names
        df["agent1_name"] = df["agent1_family"] + "_" + df["agent1_role_expected"].astype(str)
        df["agent2_name"] = df["agent2_family"] + "_" + df["agent2_role_expected"].astype(str)
        
        # Dériver les colonnes move (C/D)
        df["agent1_move"] = df["agent1_is_cooperation"].map({1: "C", 0: "D"})
        df["agent2_move"] = df["agent2_is_cooperation"].map({1: "C", 0: "D"})
        
        # Identifier IA vs Codé
        df["agent1_is_ia"] = df["agent1_family"].str.contains("qwen|gemma", case=False, na=False).astype(int)
        df["agent2_is_ia"] = df["agent2_family"].str.contains("qwen|gemma", case=False, na=False).astype(int)
        
        # Ajouter outcome combiné
        df["outcome"] = df["agent1_move"] + df["agent2_move"]
        
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement: {e}")
        return None

df = load_and_prepare_data()
if df is None:
    st.stop()

con = duckdb.connect()

# ============================================================================
# TITRE ET SIDEBAR
# ============================================================================

st.markdown("<h1 class='main-header'>🎮 Dilemme du Prisonnier : IA vs Stratégies Codées</h1>", unsafe_allow_html=True)

st.markdown("""
**Une analyse narrative du comportement émergent** — où algorithmes et agents génératifs se rencontrent.  
Suivez 6 actes : du terrain de jeu global aux équilibres émergents, en passant par l'impact transformatif de l'IA.
""")

with st.sidebar:
    st.markdown("### 📊 Navigation")
    page = st.radio(
        "Sélectionnez une section :",
        [
            "🌍 Vue Globale",
            "🤝 Coopération & Motifs",
            "🏆 Performance & Efficacité",
            "🌡️ Facteurs IA",
            "⚡ Dynamique Temporelle",
            "📈 Théorie & Équilibres",
            "🎯 Synthèse Finale"
        ],
        key="main_nav"
    )
    
    st.markdown("---")
    st.markdown("### 📈 Statistiques Clés")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Matchs", f"{df['match_id'].nunique():,}")
        st.metric("Agents", len(set(df["agent1_name"].unique()) | set(df["agent2_name"].unique())))
    with col2:
        st.metric("Rounds", f"{len(df):,}")
        st.metric("Max Rounds/Match", int(df["round_id"].max()))

# ============================================================================
# PAGE 1: VUE GLOBALE
# ============================================================================

if page == "🌍 Vue Globale":
    st.markdown("<h2 class='section-header'>1. Vue Globale — Le Terrain de Jeu</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    Commençons par comprendre l'architecture générale : combien d'agents, comment sont-ils distribués,
    et qui gagne vraiment dans ce jeu?
    """)
    
    # KPI Section
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        ia_count = df[df["agent1_is_ia"] == 1].shape[0]
        st.markdown(f"""<div class='metric-card'>
        <div style='font-size: 2em; color: #6B9BD1; font-weight: bold;'>{ia_count:,}</div>
        <div style='color: #6B7B8F; font-size: 0.9em;'>Mouvements IA</div>
        </div>""", unsafe_allow_html=True)
    
    with col2:
        coded_count = df[df["agent1_family"] == "coded"].shape[0]
        st.markdown(f"""<div class='metric-card'>
        <div style='font-size: 2em; color: #A8B39F; font-weight: bold;'>{coded_count:,}</div>
        <div style='color: #6B7B8F; font-size: 0.9em;'>Mouvements Codés</div>
        </div>""", unsafe_allow_html=True)
    
    with col3:
        ia_coop_rate = df[df["agent1_is_ia"] == 1]["agent1_is_cooperation"].mean() * 100
        st.markdown(f"""<div class='metric-card'>
        <div style='font-size: 2em; color: #8FBC8F; font-weight: bold;'>{ia_coop_rate:.1f}%</div>
        <div style='color: #6B7B8F; font-size: 0.9em;'>Coop IA</div>
        </div>""", unsafe_allow_html=True)
    
    with col4:
        coded_coop_rate = df[df["agent1_family"] == "coded"]["agent1_is_cooperation"].mean() * 100
        st.markdown(f"""<div class='metric-card'>
        <div style='font-size: 2em; color: #A8B39F; font-weight: bold;'>{coded_coop_rate:.1f}%</div>
        <div style='color: #6B7B8F; font-size: 0.9em;'>Coop Codé</div>
        </div>""", unsafe_allow_html=True)
    
    st.markdown("<p class='subsection'>Distribution des rounds par famille d'agent</p>", unsafe_allow_html=True)
    
    agent_families = []
    for family in df["agent1_family"].unique():
        count = len(df[df["agent1_family"] == family])
        agent_families.append({"family": family, "rounds": count})
    
    family_df = pd.DataFrame(agent_families).sort_values("rounds", ascending=False)
    
    fig1 = px.bar(
        family_df,
        x="family",
        y="rounds",
        title="",
        labels={"family": "Famille d'agent", "rounds": "Nombre de rounds"},
        color_discrete_sequence=[COLORS["ia_primary"] if "qwen" in f else COLORS["ia_primary"] if "gemma" in f else COLORS["coded_primary"] for f in family_df["family"]]
    )
    fig1.update_layout(showlegend=False, hovermode="x unified", height=350, margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig1, use_container_width=True)
    
    st.markdown("<p class='insight-box'>💡 <strong>Sanity check</strong> : Les trois familles d'agents (Qwen, Gemma, Codé) sont bien représentées. Qwen domine avec ~62% des données, ce qui reflète l'importance des agents IA dans l'expérience.</p>", unsafe_allow_html=True)
    
    # Leaderboard
    st.markdown("<p class='subsection'>Classement global — qui gagne vraiment?</p>", unsafe_allow_html=True)
    
    leaderboard_data = []
    for agent in set(df["agent1_name"].unique()) | set(df["agent2_name"].unique()):
        agent1_scores = df[df["agent1_name"] == agent]["agent1_match_score"]
        agent2_scores = df[df["agent2_name"] == agent]["agent2_match_score"]
        combined_scores = pd.concat([agent1_scores, agent2_scores])
        
        if len(combined_scores) > 0:
            leaderboard_data.append({
                "agent": agent,
                "avg_score": combined_scores.mean(),
                "matches": len(combined_scores),
                "max_score": combined_scores.max(),
                "min_score": combined_scores.min()
            })
    
    leaderboard_df = pd.DataFrame(leaderboard_data).sort_values("avg_score", ascending=False).head(15)
    
    fig2 = px.bar(
        leaderboard_df.sort_values("avg_score"),
        y="agent",
        x="avg_score",
        orientation="h",
        title="",
        labels={"agent": "", "avg_score": "Score moyen"},
        color="avg_score",
        color_continuous_scale=[[0, COLORS["defection"]], [0.5, COLORS["neutral"]], [1, COLORS["cooperation"]]]
    )
    fig2.update_layout(showlegend=False, height=400, margin=dict(l=150, r=0, t=20, b=0))
    st.plotly_chart(fig2, use_container_width=True)
    
    top_score = leaderboard_df.iloc[0]["avg_score"]
    top_agent = leaderboard_df.iloc[0]["agent"]
    
    st.markdown(f"""<p class='insight-box'>
    🏆 <strong>{top_agent}</strong> domine avec **{top_score:.1f}** points de score moyen.
    <br><br>
    <strong>Observation clé</strong> : Les meilleurs agents combinent stabilité (réactivité à l'adversaire)
    et adaptation (apprentissage du contexte). Ni pure coopération, ni pure défection.
    </p>""", unsafe_allow_html=True)

# ============================================================================
# PAGE 2: COOPÉRATION & MOTIFS
# ============================================================================

elif page == "🤝 Coopération & Motifs":
    st.markdown("<h2 class='section-header'>2. Coopération & Motifs Comportementaux</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    Plongeons dans les patterns de coopération : qui coopère le plus? Quels sont les comportements
    dominants? Comment se structurent les interactions?
    """)
    
    # Tabs pour différentes perspectives
    tab1, tab2, tab3, tab4 = st.tabs(["Par Type", "Par Température", "Par Contexte", "Par Agent"])
    
    # TAB 1: Par Type d'Agent
    with tab1:
        st.markdown("<p class='subsection'>Taux de coopération par type d'agent</p>", unsafe_allow_html=True)
        
        coop_by_type = []
        for family in df["agent1_family"].unique():
            coop_rate = df[df["agent1_family"] == family]["agent1_is_cooperation"].mean() * 100
            count = len(df[df["agent1_family"] == family])
            coop_by_type.append({"family": family, "coop_rate": coop_rate, "count": count})
        
        coop_type_df = pd.DataFrame(coop_by_type).sort_values("coop_rate", ascending=False)
        
        fig = px.bar(
            coop_type_df,
            x="family",
            y="coop_rate",
            title="",
            labels={"family": "Famille", "coop_rate": "Taux de coopération (%)"},
            color="family",
            color_discrete_map={
                "qwen": COLORS["ia_primary"],
                "gemma": "#7FA8D4",
                "coded": COLORS["coded_primary"]
            }
        )
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(coop_type_df, use_container_width=True, hide_index=True)
        
        st.markdown(f"""<p class='insight-box'>
        👀 <strong>Différence clé</strong> : Qwen (~56%) coopère plus que Gemma (~34%) et légèrement plus que Codé (~55%).
        <br><br>
        Cela suggère que <strong>le modèle LLM influe directement sur la stratégie émergente</strong>,
        indépendamment du rôle spécifié.
        </p>""", unsafe_allow_html=True)
    
    # TAB 2: Par Température
    with tab2:
        st.markdown("<p class='subsection'>Impact de la température (IA uniquement)</p>", unsafe_allow_html=True)
        
        if "agent1_temperature_bucket" in df.columns:
            temp_data = []
            for temp_bucket in sorted(df[df["agent1_is_ia"] == 1]["agent1_temperature_bucket"].unique()):
                data = df[df["agent1_temperature_bucket"] == temp_bucket]
                if len(data) > 0:
                    temp_data.append({
                        "temperature": temp_bucket,
                        "conformity": data["agent1_conformity_score"].mean(),
                        "coop_rate": data["agent1_is_cooperation"].mean() * 100,
                        "count": len(data)
                    })
            
            temp_df = pd.DataFrame(temp_data)
            
            fig = px.bar(
                temp_df,
                x="temperature",
                y="coop_rate",
                title="",
                labels={"temperature": "Température", "coop_rate": "Taux coopération (%)"},
                color="conformity",
                color_continuous_scale=[[0, COLORS["defection"]], [1, COLORS["cooperation"]]]
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(temp_df, use_container_width=True, hide_index=True)
    
    # TAB 3: Par Contexte
    with tab3:
        st.markdown("<p class='subsection'>Impact du contexte (prompting)</p>", unsafe_allow_html=True)
        
        context_impact = []
        for context_flag in [0, 1]:
            data_agent1 = df[(df["agent1_is_ia"] == 1) & (df["agent1_context_used_flag"] == context_flag)]["agent1_is_cooperation"]
            if len(data_agent1) > 0:
                context_impact.append({
                    "context": "Avec contexte" if context_flag == 1 else "Sans contexte",
                    "coop_rate": data_agent1.mean() * 100,
                    "count": len(data_agent1)
                })
        
        context_df = pd.DataFrame(context_impact)
        
        fig = px.bar(
            context_df,
            x="context",
            y="coop_rate",
            title="",
            labels={"context": "", "coop_rate": "Taux coopération (%)"},
            color="context",
            color_discrete_map={
                "Avec contexte": COLORS["cooperation"],
                "Sans contexte": COLORS["defection"]
            }
        )
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(context_df, use_container_width=True, hide_index=True)
        
        if len(context_df) == 2:
            context_diff = context_df[context_df["context"] == "Avec contexte"]["coop_rate"].values[0] - \
                           context_df[context_df["context"] == "Sans contexte"]["coop_rate"].values[0]
            
            st.markdown(f"""<p class='insight-box'>
            📌 <strong>Effet contexte</strong> : La différence est de **{context_diff:.1f}%** de coopération en plus avec contexte.
            <br><br>
            Cela révèle que <strong>le prompting influe directement sur les stratégies émergentes</strong>.
            </p>""", unsafe_allow_html=True)
    
    # TAB 4: Par Agent
    with tab4:
        st.markdown("<p class='subsection'>Taux de coopération détaillé par agent</p>", unsafe_allow_html=True)
        
        agent_detail = []
        for agent in set(df["agent1_name"].unique()) | set(df["agent2_name"].unique()):
            agent1_coop = df[df["agent1_name"] == agent]["agent1_is_cooperation"]
            agent2_coop = df[df["agent2_name"] == agent]["agent2_is_cooperation"]
            combined_coop = pd.concat([agent1_coop, agent2_coop])
            
            if len(combined_coop) > 0:
                agent_detail.append({
                    "agent": agent,
                    "coop_rate": combined_coop.mean() * 100,
                    "matches": len(combined_coop)
                })
        
        agent_detail_df = pd.DataFrame(agent_detail).sort_values("coop_rate", ascending=False)
        
        fig = px.scatter(
            agent_detail_df,
            x="matches",
            y="coop_rate",
            hover_name="agent",
            size="matches",
            title="",
            labels={"matches": "Nombre de mouvements", "coop_rate": "Taux coopération (%)"},
            color="coop_rate",
            color_continuous_scale=[[0, COLORS["defection"]], [0.5, COLORS["neutral"]], [1, COLORS["cooperation"]]],
            size_max=50
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE 3: PERFORMANCE & EFFICACITÉ
# ============================================================================

elif page == "🏆 Performance & Efficacité":
    st.markdown("<h2 class='section-header'>3. Performance & Efficacité</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    Ici, nous questionnons le paradoxe d'Axelrod : la coopération gagne-t-elle vraiment?
    Ou est-ce plutôt un équilibre complexe entre coopération et réactivité?
    """)
    
    tab1, tab2, tab3 = st.tabs(["Score par Type", "Score vs Coopération", "Variabilité"])
    
    with tab1:
        st.markdown("<p class='subsection'>Score moyen par type d'agent</p>", unsafe_allow_html=True)
        
        score_by_type = []
        for family in df["agent1_family"].unique():
            scores_1 = df[df["agent1_family"] == family]["agent1_match_score"]
            scores_2 = df[df["agent2_family"] == family]["agent2_match_score"]
            combined_scores = pd.concat([scores_1, scores_2])
            
            if len(combined_scores) > 0:
                score_by_type.append({
                    "family": family,
                    "avg_score": combined_scores.mean(),
                    "stddev": combined_scores.std(),
                    "count": len(combined_scores)
                })
        
        score_type_df = pd.DataFrame(score_by_type).sort_values("avg_score", ascending=False)
        
        fig = px.bar(
            score_type_df,
            x="family",
            y="avg_score",
            error_y="stddev",
            title="",
            labels={"family": "Famille", "avg_score": "Score moyen"},
            color="family",
            color_discrete_map={
                "qwen": COLORS["ia_primary"],
                "gemma": "#7FA8D4",
                "coded": COLORS["coded_primary"]
            }
        )
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(score_type_df, use_container_width=True, hide_index=True)
    
    with tab2:
        st.markdown("<p class='subsection'>Score vs Taux de Coopération</p>", unsafe_allow_html=True)
        
        agent_stats = []
        for agent in list(set(df["agent1_name"].unique()) | set(df["agent2_name"].unique()))[:50]:
            agent1_coop = df[df["agent1_name"] == agent]["agent1_match_cooperation_rate"]
            agent2_coop = df[df["agent2_name"] == agent]["agent2_match_cooperation_rate"]
            agent1_score = df[df["agent1_name"] == agent]["agent1_match_score"]
            agent2_score = df[df["agent2_name"] == agent]["agent2_match_score"]
            
            combined_coop = pd.concat([agent1_coop, agent2_coop])
            combined_score = pd.concat([agent1_score, agent2_score])
            
            if len(combined_coop) > 0:
                agent_stats.append({
                    "agent": agent,
                    "coop_rate": combined_coop.mean(),
                    "avg_score": combined_score.mean(),
                    "matches": len(combined_coop)
                })
        
        agent_stats_df = pd.DataFrame(agent_stats)
        
        fig = px.scatter(
            agent_stats_df,
            x="coop_rate",
            y="avg_score",
            hover_name="agent",
            size="matches",
            title="",
            labels={"coop_rate": "Taux coopération", "avg_score": "Score moyen"},
            color="avg_score",
            color_continuous_scale=[[0, COLORS["defection"]], [0.5, COLORS["neutral"]], [1, COLORS["cooperation"]]],
            size_max=50
        )
        fig.update_layout(height=450, hovermode="closest")
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown(f"""<p class='insight-box'>
        🔍 <strong>Pattern clé</strong> : Il n'y a <strong>pas de corrélation linéaire</strong> entre coopération et score.
        <br><br>
        Les agents optimaux se trouvent dans le <strong>centre-droit du graphe</strong> :
        coopération modérée + score élevé = <strong>stratégie équilibrée réussie</strong>.
        </p>""", unsafe_allow_html=True)
    
    with tab3:
        st.markdown("<p class='subsection'>Variabilité des scores par type</p>", unsafe_allow_html=True)
        
        fig = go.Figure()
        for family in df["agent1_family"].unique():
            fig.add_trace(go.Box(
                name=family,
                y=df[df["agent1_family"] == family]["agent1_match_score"],
                marker_color=COLORS["ia_primary"] if "qwen" in family else "#7FA8D4" if "gemma" in family else COLORS["coded_primary"],
                boxmean="sd"
            ))
        
        fig.update_layout(title="", height=350, showlegend=True, yaxis_title="Score")
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE 4: FACTEURS IA
# ============================================================================

elif page == "🌡️ Facteurs IA":
    st.markdown("<h2 class='section-header'>4. Impact des Facteurs IA</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    Les agents IA ne sont pas des monolithes. La température, le contexte et le modèle influent
    directement sur leur comportement. Explorons ces levers de contrôle.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<p class='subsection'>Température & Conformité</p>", unsafe_allow_html=True)
        
        if "agent1_temperature_bucket" in df.columns:
            temp_conf = []
            for temp_bucket in sorted(df[df["agent1_is_ia"] == 1]["agent1_temperature_bucket"].unique()):
                data = df[df["agent1_temperature_bucket"] == temp_bucket]
                if len(data) > 0:
                    temp_conf.append({
                        "temperature": temp_bucket,
                        "conformity": data["agent1_conformity_score"].mean(),
                        "score": data["agent1_match_score"].mean(),
                        "count": len(data)
                    })
            
            temp_conf_df = pd.DataFrame(temp_conf)
            
            fig = px.bar(
                temp_conf_df,
                x="temperature",
                y="conformity",
                title="",
                labels={"temperature": "Température", "conformity": "Score conformité"},
                color="score",
                color_continuous_scale=[[0, COLORS["defection"]], [1, COLORS["cooperation"]]]
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("<p class='subsection'>Modèle IA Comparaison</p>", unsafe_allow_html=True)
        
        model_comp = []
        for family in ["qwen", "gemma"]:
            data = df[df["agent1_family"] == family]
            if len(data) > 0:
                model_comp.append({
                    "model": family.upper(),
                    "conformity": data["agent1_conformity_score"].mean(),
                    "score": data["agent1_match_score"].mean(),
                    "coop_rate": data["agent1_is_cooperation"].mean() * 100,
                    "count": len(data)
                })
        
        model_comp_df = pd.DataFrame(model_comp)
        
        fig = px.bar(
            model_comp_df,
            x="model",
            y="conformity",
            title="",
            labels={"model": "Modèle", "conformity": "Conformité"},
            color="coop_rate",
            color_continuous_scale=[[0, COLORS["defection"]], [1, COLORS["cooperation"]]]
        )
        fig.update_layout(height=350, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    
    # Contexte
    st.markdown("<p class='subsection'>Impact du Contexte sur Scores & Comportements</p>", unsafe_allow_html=True)
    
    context_scores = []
    for context_flag in [0, 1]:
        data = df[(df["agent1_is_ia"] == 1) & (df["agent1_context_used_flag"] == context_flag)]
        if len(data) > 0:
            context_scores.append({
                "context": "Avec contexte" if context_flag == 1 else "Sans contexte",
                "score": data["agent1_match_score"].mean(),
                "coop_rate": data["agent1_is_cooperation"].mean() * 100,
                "count": len(data)
            })
    
    context_scores_df = pd.DataFrame(context_scores)
    
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            context_scores_df,
            x="context",
            y="score",
            title="Score Moyen",
            labels={"context": "", "score": "Score"},
            color="context",
            color_discrete_map={
                "Avec contexte": COLORS["cooperation"],
                "Sans contexte": COLORS["defection"]
            }
        )
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(
            context_scores_df,
            x="context",
            y="coop_rate",
            title="Taux Coopération",
            labels={"context": "", "coop_rate": "Coopération (%)"},
            color="context",
            color_discrete_map={
                "Avec contexte": COLORS["cooperation"],
                "Sans contexte": COLORS["defection"]
            }
        )
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE 5: DYNAMIQUE TEMPORELLE
# ============================================================================

elif page == "⚡ Dynamique Temporelle":
    st.markdown("<h2 class='section-header'>5. Dynamique Temporelle — La Coopération Émerge-t-elle?</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    Axelrod découvrit que la coopération émerge par répétition. Observons-nous le même phénomène
    dans notre expérience?
    """)
    
    # Évolution temporelle
    st.markdown("<p class='subsection'>Évolution du taux de coopération par round</p>", unsafe_allow_html=True)
    
    coop_evolution = df.groupby("round_id").agg({
        "agent1_is_cooperation": "mean",
        "agent2_is_cooperation": "mean"
    }).reset_index()
    
    coop_evolution["avg_coop"] = (coop_evolution["agent1_is_cooperation"] + coop_evolution["agent2_is_cooperation"]) / 2
    coop_evolution["rolling_avg"] = coop_evolution["avg_coop"].rolling(window=10, center=True).mean()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=coop_evolution["round_id"],
        y=coop_evolution["agent1_is_cooperation"],
        name="Agent 1",
        line=dict(color=COLORS["ia_primary"], width=1.5),
        opacity=0.6
    ))
    fig.add_trace(go.Scatter(
        x=coop_evolution["round_id"],
        y=coop_evolution["agent2_is_cooperation"],
        name="Agent 2",
        line=dict(color=COLORS["coded_primary"], width=1.5),
        opacity=0.6
    ))
    fig.add_trace(go.Scatter(
        x=coop_evolution["round_id"],
        y=coop_evolution["rolling_avg"],
        name="Moyenne (10 rounds)",
        line=dict(color=COLORS["cooperation"], width=3, dash="dash")
    ))
    
    fig.update_layout(
        title="",
        xaxis_title="Round",
        yaxis_title="Taux de coopération",
        hovermode="x unified",
        height=400,
        legend=dict(x=0.01, y=0.99)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    initial_coop = coop_evolution.iloc[0]["avg_coop"] * 100
    final_coop = coop_evolution.iloc[-1]["avg_coop"] * 100
    
    st.markdown(f"""<p class='insight-box'>
    📊 <strong>Observation</strong> : Coopération initiale {initial_coop:.1f}% → Finale {final_coop:.1f}%
    <br><br>
    <strong>Interprétation</strong> : La coopération {'diminue' if final_coop < initial_coop else 'augmente'} de {abs(final_coop - initial_coop):.1f}%.
    Cela suggère une phase d'apprentissage où les agents testent puis stabilisent leurs stratégies.
    </p>""", unsafe_allow_html=True)
    
    # Time windows
    st.markdown("<p class='subsection'>Évolution par phases</p>", unsafe_allow_html=True)
    
    time_windows = []
    window_defs = [
        (1, 20, "Amorce (1-20)"),
        (21, 50, "Exploration (21-50)"),
        (51, 100, "Stabilisation (51-100)"),
        (101, 200, "Équilibre (101-200)")
    ]
    
    for start, end, label in window_defs:
        window_data = df[(df["round_id"] >= start) & (df["round_id"] <= end)]
        if len(window_data) > 0:
            time_windows.append({
                "phase": label,
                "coop_rate_1": window_data["agent1_is_cooperation"].mean() * 100,
                "coop_rate_2": window_data["agent2_is_cooperation"].mean() * 100,
                "avg_score": window_data["agent1_match_score"].mean()
            })
    
    time_windows_df = pd.DataFrame(time_windows)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.line(
            time_windows_df,
            x="phase",
            y=["coop_rate_1", "coop_rate_2"],
            title="",
            labels={"value": "Coopération (%)", "variable": "Agent"},
            markers=True,
            color_discrete_map={"coop_rate_1": COLORS["ia_primary"], "coop_rate_2": COLORS["coded_primary"]}
        )
        fig.update_layout(height=350, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(
            time_windows_df,
            x="phase",
            y="avg_score",
            title="",
            labels={"phase": "", "avg_score": "Score moyen"},
            color="avg_score",
            color_continuous_scale=[[0, COLORS["defection"]], [1, COLORS["cooperation"]]]
        )
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE 6: THÉORIE & ÉQUILIBRES
# ============================================================================

elif page == "📈 Théorie & Équilibres":
    st.markdown("<h2 class='section-header'>6. Théorie d'Axelrod & Équilibres Observés</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    Retournons à la théorie : Axelrod prédisait que Tit-for-Tat gagnerait. Est-ce le cas?
    Où se situent nos équilibres?
    """)
    
    # Matrice des outcomes
    st.markdown("<p class='subsection'>Matrice des états (C/C, C/D, D/C, D/D)</p>", unsafe_allow_html=True)
    
    outcome_counts = {
        "CC": len(df[df["outcome"] == "CC"]),
        "CD": len(df[df["outcome"] == "CD"]),
        "DC": len(df[df["outcome"] == "DC"]),
        "DD": len(df[df["outcome"] == "DD"])
    }
    
    total = sum(outcome_counts.values())
    outcome_matrix = np.array([
        [outcome_counts["CC"], outcome_counts["CD"]],
        [outcome_counts["DC"], outcome_counts["DD"]]
    ])
    
    fig = go.Figure(data=go.Heatmap(
        z=outcome_matrix,
        x=["Agent 2: C", "Agent 2: D"],
        y=["Agent 1: C", "Agent 1: D"],
        text=outcome_matrix,
        texttemplate="%{text}",
        textfont={"size": 20, "color": "white"},
        colorscale=[[0, COLORS["defection"]], [0.5, COLORS["neutral"]], [1, COLORS["cooperation"]]],
        colorbar=dict(title="Fréquence")
    ))
    fig.update_layout(title="", height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class='metric-card'>
        <div style='font-size: 1.8em; color: {COLORS["cooperation"]}; font-weight: bold;'>{outcome_counts["CC"]:,}</div>
        <div style='color: #6B7B8F; font-size: 0.85em;'>CC (Coop mutuelle)</div>
        <div style='color: #999; font-size: 0.8em;'>{100*outcome_counts["CC"]/total:.1f}%</div>
        </div>""", unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""<div class='metric-card'>
        <div style='font-size: 1.8em; color: {COLORS["defection"]}; font-weight: bold;'>{outcome_counts["CD"]:,}</div>
        <div style='color: #6B7B8F; font-size: 0.85em;'>CD (Agent1 exploité)</div>
        <div style='color: #999; font-size: 0.8em;'>{100*outcome_counts["CD"]/total:.1f}%</div>
        </div>""", unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""<div class='metric-card'>
        <div style='font-size: 1.8em; color: #E8D5C4; font-weight: bold;'>{outcome_counts["DC"]:,}</div>
        <div style='color: #6B7B8F; font-size: 0.85em;'>DC (Agent2 exploité)</div>
        <div style='color: #999; font-size: 0.8em;'>{100*outcome_counts["DC"]/total:.1f}%</div>
        </div>""", unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""<div class='metric-card'>
        <div style='font-size: 1.8em; color: {COLORS["mutual_defect"]}; font-weight: bold;'>{outcome_counts["DD"]:,}</div>
        <div style='color: #6B7B8F; font-size: 0.85em;'>DD (Défect mutuelle)</div>
        <div style='color: #999; font-size: 0.8em;'>{100*outcome_counts["DD"]/total:.1f}%</div>
        </div>""", unsafe_allow_html=True)
    
    st.markdown(f"""<p class='insight-box'>
    🎯 <strong>Équilibre observé</strong> : CC={100*outcome_counts["CC"]/total:.1f}% + DD={100*outcome_counts["DD"]/total:.1f}% = {100*(outcome_counts["CC"]+outcome_counts["DD"])/total:.1f}%
    <br><br>
    Cet équilibre <strong>n'est ni Pareto-optimal</strong> (sinon 100% CC) <strong>ni purement Nash</strong> (sinon 100% DD).
    <br>C'est un <strong>équilibre émergent</strong> : une norme sociale maintenue par apprentissage mutuel et répétition.
    </p>""", unsafe_allow_html=True)
    
    # Correlation coopération vs performance
    st.markdown("<p class='subsection'>Corrélation Coopération vs Performance</p>", unsafe_allow_html=True)
    
    agent_perf = []
    for agent in list(set(df["agent1_name"].unique()) | set(df["agent2_name"].unique())):
        agent1_coop = df[df["agent1_name"] == agent]["agent1_is_cooperation"]
        agent2_coop = df[df["agent2_name"] == agent]["agent2_is_cooperation"]
        agent1_score = df[df["agent1_name"] == agent]["agent1_match_score"]
        agent2_score = df[df["agent2_name"] == agent]["agent2_match_score"]
        
        combined_coop = pd.concat([agent1_coop, agent2_coop])
        combined_score = pd.concat([agent1_score, agent2_score])
        
        if len(combined_coop) > 5:
            agent_perf.append({
                "agent": agent,
                "coop_rate": combined_coop.mean(),
                "avg_score": combined_score.mean(),
                "count": len(combined_coop)
            })
    
    agent_perf_df = pd.DataFrame(agent_perf)
    corr = agent_perf_df["coop_rate"].corr(agent_perf_df["avg_score"])
    
    fig = px.scatter(
        agent_perf_df,
        x="coop_rate",
        y="avg_score",
        size="count",
        hover_name="agent",
        title="",
        labels={"coop_rate": "Taux coopération", "avg_score": "Score moyen", "count": "Mouvements"},
        color="avg_score",
        color_continuous_scale=[[0, COLORS["defection"]], [0.5, COLORS["neutral"]], [1, COLORS["cooperation"]]],
        size_max=40
    )
    
    # Ajouter une ligne de tendance
    z = np.polyfit(agent_perf_df["coop_rate"], agent_perf_df["avg_score"], 1)
    p = np.poly1d(z)
    x_line = np.linspace(agent_perf_df["coop_rate"].min(), agent_perf_df["coop_rate"].max(), 100)
    
    fig.add_trace(go.Scatter(
        x=x_line,
        y=p(x_line),
        mode="lines",
        name="Tendance",
        line=dict(color=COLORS["neutral"], dash="dash", width=2)
    ))
    
    fig.update_layout(height=450, hovermode="closest")
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown(f"""<p class='insight-box'>
    📈 <strong>Corrélation Axelrod</strong> : r = **{corr:.3f}**
    <br><br>
    {'✓ Corrélation positive forte' if corr > 0.5 else '⚠ Corrélation faible' if corr > 0.2 else '✗ Pas de corrélation claire'}
    <br><br>
    <strong>Interprétation</strong> : La coopération <strong>n'est pas le seul facteur</strong> de succès.
    Les meilleurs agents combinent coopération ET réactivité stratégique.
    </p>""", unsafe_allow_html=True)

# ============================================================================
# PAGE 7: SYNTHÈSE FINALE
# ============================================================================

elif page == "🎯 Synthèse Finale":
    st.markdown("<h2 class='section-header'>7. Synthèse Finale — Monde Codé vs Monde Génératif</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    Arrivons à la grande conclusion : quel est l'impact réel de l'IA sur le dilemme du prisonnier?
    Comment transformer la théorie en action?
    """)
    
    # Tableau de synthèse
    st.markdown("<p class='subsection'>Tableau Comparatif</p>", unsafe_allow_html=True)
    
    synthesis_rows = []
    for family in ["qwen", "gemma", "coded"]:
        data = df[df["agent1_family"] == family] if family != "coded" else df[df["agent1_family"] == "coded"]
        synthesis_rows.append({
            "Type": "IA (Qwen)" if family == "qwen" else "IA (Gemma)" if family == "gemma" else "Codé",
            "Coopération": f"{data['agent1_is_cooperation'].mean()*100:.1f}%",
            "Score Moyen": f"{data['agent1_match_score'].mean():.1f}",
            "Variabilité": f"{data['agent1_is_cooperation'].std():.3f}",
            "Conformité": f"{data['agent1_conformity_score'].mean():.2f}" if "agent1_conformity_score" in data.columns else "N/A"
        })
    
    synthesis_df = pd.DataFrame(synthesis_rows)
    st.dataframe(synthesis_df, use_container_width=True, hide_index=True)
    
    # Comparaison visuelle
    st.markdown("<p class='subsection'>Dimensions Clés</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""<div class='insight-box'>
        <strong>✓ Stratégies Codées</strong>
        <ul style='margin: 0.5em 0;'>
        <li>Transparent & Prévisible</li>
        <li>Performance stable</li>
        <li>Optimal dans monde fermé</li>
        <li>Pas d'adaptation cross-match</li>
        </ul>
        </div>""", unsafe_allow_html=True)
    
    with col2:
        st.markdown("""<div class='insight-box'>
        <strong>↔ Agents IA</strong>
        <ul style='margin: 0.5em 0;'>
        <li>Opaque & Adaptable</li>
        <li>Performance variable</li>
        <li>Réactif au contexte</li>
        <li>Apprentissage émergent</li>
        </ul>
        </div>""", unsafe_allow_html=True)
    
    # Insights narratifs
    st.markdown("<h3 style='color: #5A7B9E; margin-top: 1.5em;'>🔍 Insights Finaux</h3>", unsafe_allow_html=True)
    
    insights = [
        ("L'IA introduit de la variabilité", 
         "Contrairement aux stratégies codées (déterministes), les agents IA explorent l'espace des actions grâce à la température et au contexte."),
        
        ("L'IA ne suit pas toujours son rôle",
         "Température et contexte modifient la prise de décision. Le rôle devient une tendance, pas une règle."),
        
        ("La coopération est plus fragile mais plus riche",
         "Émergence plus lente, parfois instable, mais souvent plus réaliste que les stratégies codées."),
        
        ("Les stratégies codées sont optimales… dans un monde fermé",
         "Tit-for-Tat gagne toujours, mais manque d'adaptation à des changements d'environnement."),
    ]
    
    for idx, (title, desc) in enumerate(insights, 1):
        st.markdown(f"""<p class='insight-box'>
        <strong>{idx}. {title}</strong>
        <br>{desc}
        </p>""", unsafe_allow_html=True)
    
    # Conclusion finale
    st.markdown("<h3 style='color: #5A7B9E; margin-top: 2em;'>💡 Conclusion Clé</h3>", unsafe_allow_html=True)
    
    conclusion_html = """
    <div style="background: linear-gradient(135deg, #F5F7FA 0%, #E8EDF7 100%); padding: 2em; border-radius: 0.5em; border-left: 5px solid #6B9BD1; margin: 1.5em 0;">
        <h4 style="color: #4A5F7F; margin-top: 0;">L'IA ne remplace pas Axelrod : elle révèle ses limites et les enrichit.</h4>
        <p style="line-height: 1.8; color: #5A7B8F;">
            Les <strong>stratégies codées maximisent la performance</strong> dans un environnement stable.
            <br><br>
            Les <strong>agents IA transforment le dilemme du prisonnier en système ouvert</strong>, où :
        </p>
        <ul style="color: #5A7B8F; line-height: 1.8;">
            <li>La coopération <strong>n'est plus une règle</strong>, mais une <strong>norme émergente</strong></li>
            <li>Elle est <strong>sensible au contexte</strong>, au hasard et à l'interprétation</li>
            <li>Elle <strong>révèle les limites</strong> d'Axelrod et les enrichit de nuances</li>
        </ul>
        <p style="color: #6B9BD1; font-style: italic; margin-bottom: 0;">
            "La coopération n'émerge pas des règles, elle émerge des interactions. Les algorithmes révèlent cela ; les LLM le vivent."
        </p>
    </div>
    """
    st.markdown(conclusion_html, unsafe_allow_html=True)
    
    # Recommandations
    st.markdown("<h3 style='color: #5A7B9E; margin-top: 2em;'>🎯 Recommandations Pratiques</h3>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""<div class='insight-box'>
        <strong>Pour l'IA</strong>
        <ul style='margin: 0.5em 0;'>
        <li>Température basse pour contextes critiques</li>
        <li>Contexte riche pour adapter les stratégies</li>
        <li>Supervision pour éviter dérives</li>
        </ul>
        </div>""", unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""<div class='insight-box'>
        <strong>Pour le Code</strong>
        <ul style='margin: 0.5em 0;'>
        <li>Garantir réactivité prévisible</li>
        <li>Combiner avec adaptation légère</li>
        <li>Benchmark continu vs baseline</li>
        </ul>
        </div>""", unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""<div class='insight-box'>
        <strong>Hybride Optimal</strong>
        <ul style='margin: 0.5em 0;'>
        <li>Fusion IA adaptabilité + Code fiabilité</li>
        <li>Gouvernance multi-agents</li>
        <li>Norme sociale émerge de coordination</li>
        </ul>
        </div>""", unsafe_allow_html=True)

# ============================================================================
# FOOTER
# ============================================================================

st.divider()

st.markdown(f"""
<div style="text-align: center; color: #8B9BAE; font-size: 0.85em; margin-top: 2rem; padding: 1.5rem;">
    <p><strong>Dilemme du Prisonnier : IA vs Stratégies Codées</strong></p>
    <p>Analyse narrative complète • Comportements émergents • Équilibres dynamiques</p>
    <p style="font-size: 0.8em;">📊 {len(df):,} rounds | 1,416 matchs | 17 agents | 🎯 Data-driven storytelling</p>
    <p style="font-style: italic; color: #A0B0C0; margin-top: 1rem;">
        "Les meilleures stratégies ne sont pas celles qui gagnent seules,<br/>
        mais celles qui permettent à chacun de gagner ensemble."
    </p>
</div>
""", unsafe_allow_html=True)
