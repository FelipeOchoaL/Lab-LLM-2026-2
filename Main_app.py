import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import tiktoken
from groq import Groq
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

st.set_page_config(page_title="Laboratorio de LLMs con Groq", page_icon="🤖", layout="wide")

DEFAULT_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
]

ENCODINGS = {
    "o200k_base (GPT-4o / GPT-OSS)": "o200k_base",
    "cl100k_base (GPT-4 / GPT-3.5)": "cl100k_base",
    "p50k_base (GPT-3 / Codex)": "p50k_base",
    "r50k_base (GPT-2)": "r50k_base",
}

# ---------------- Sidebar ----------------
st.sidebar.title("🔑 Configuración")
api_key = st.sidebar.text_input("API Key de GROQ", type="password")
client = None
models = DEFAULT_MODELS

if api_key:
    try:
        client = Groq(api_key=api_key)
        available = sorted(m.id for m in client.models.list().data)
        models = [m for m in available if "gpt-oss" in m.lower() and "safeguard" not in m.lower()] or DEFAULT_MODELS
        st.sidebar.success(f"Conectado: {len(available)} modelos disponibles")
    except Exception as e:
        st.sidebar.error(f"Error con la API key: {e}")
        client = None
else:
    st.sidebar.info("Ingresa tu API key para habilitar la generación de texto.")

st.title("🤖 Laboratorio de LLMs")
st.caption("Tokens, IDs, Bag of Words, similitud, embeddings y generación de texto con Groq.")

tabs = st.tabs(["🧩 Tokens", "👜 Bag of Words", "📏 Similitud", "🧭 Embeddings", "✍️ Generación", "⚖️ Comparar modelos"])

# ---------------- Tokens ----------------
with tabs[0]:
    st.subheader("Tokenización")
    text = st.text_area("Texto", "Los modelos de lenguaje convierten el texto en tokens.", key="tok_text")
    selected = st.multiselect("Tokenizadores", list(ENCODINGS), default=list(ENCODINGS)[:2])
    for name in selected:
        enc = tiktoken.get_encoding(ENCODINGS[name])
        ids = enc.encode(text)
        pieces = [enc.decode_single_token_bytes(i).decode("utf-8", errors="replace") for i in ids]
        st.markdown(f"**{name}** — {len(ids)} tokens, {len(text)} caracteres")
        colors = ["#FFD6A5", "#CAFFBF", "#9BF6FF", "#BDB2FF", "#FFC6FF"]
        html = "".join(
            f"<span style='background:{colors[i % len(colors)]};padding:2px;margin:1px;border-radius:3px;color:black'>"
            f"{p.replace(' ', '&nbsp;')}</span>" for i, p in enumerate(pieces)
        )
        st.markdown(html, unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({"Token": pieces, "Token ID": ids}), use_container_width=True, height=200)

# ---------------- Bag of Words ----------------
with tabs[1]:
    st.subheader("Bag of Words y TF-IDF")
    docs_raw = st.text_area(
        "Documentos (uno por línea)",
        "El gato duerme en la casa\nEl perro juega en el parque\nEl gato y el perro son amigos",
        key="bow_docs",
    )
    docs = [d for d in docs_raw.splitlines() if d.strip()]
    ngram = st.slider("Rango de n-gramas (máx)", 1, 3, 1)
    if docs:
        c1, c2 = st.columns(2)
        cv = CountVectorizer(ngram_range=(1, ngram))
        bow = cv.fit_transform(docs)
        bow_df = pd.DataFrame(bow.toarray(), columns=cv.get_feature_names_out(), index=[f"Doc {i+1}" for i in range(len(docs))])
        c1.markdown("**Bag of Words (conteos)**")
        c1.dataframe(bow_df, use_container_width=True)
        tv = TfidfVectorizer(ngram_range=(1, ngram))
        tfidf = tv.fit_transform(docs)
        tf_df = pd.DataFrame(tfidf.toarray().round(3), columns=tv.get_feature_names_out(), index=bow_df.index)
        c2.markdown("**TF-IDF**")
        c2.dataframe(tf_df, use_container_width=True)
        freq = bow_df.sum().sort_values(ascending=False).head(20)
        st.plotly_chart(px.bar(x=freq.index, y=freq.values, labels={"x": "Término", "y": "Frecuencia"}), use_container_width=True)

# ---------------- Similitud ----------------
with tabs[2]:
    st.subheader("Métricas de similitud")
    a = st.text_input("Texto A", "Me gusta programar en Python")
    b = st.text_input("Texto B", "Disfruto escribir código en Python")
    if a and b:
        tv = TfidfVectorizer().fit([a, b])
        va, vb = tv.transform([a]), tv.transform([b])
        sa, sb = set(a.lower().split()), set(b.lower().split())
        enc = tiktoken.get_encoding("o200k_base")
        ta, tb = set(enc.encode(a)), set(enc.encode(b))
        metrics = {
            "Coseno (TF-IDF)": float(cosine_similarity(va, vb)[0, 0]),
            "Distancia euclidiana (TF-IDF)": float(euclidean_distances(va, vb)[0, 0]),
            "Jaccard (palabras)": len(sa & sb) / len(sa | sb),
            "Jaccard (tokens)": len(ta & tb) / len(ta | tb),
        }
        cols = st.columns(len(metrics))
        for col, (k, v) in zip(cols, metrics.items()):
            col.metric(k, f"{v:.3f}")
        st.caption("Coseno y Jaccard: 1 = idénticos. Distancia euclidiana: 0 = idénticos.")

# ---------------- Embeddings ----------------
with tabs[3]:
    st.subheader("Embeddings (TF-IDF + PCA) y matriz de similitud")
    st.caption("Groq no ofrece endpoint de embeddings; se usan vectores TF-IDF de caracteres como embeddings locales.")
    emb_raw = st.text_area(
        "Frases (una por línea)",
        "El rey gobierna el reino\nLa reina gobierna el reino\nEl perro ladra\nEl gato maúlla\nPython es un lenguaje de programación\nJavaScript se usa en la web",
        key="emb",
    )
    sents = [s for s in emb_raw.splitlines() if s.strip()]
    if len(sents) >= 3:
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
        X = vec.fit_transform(sents).toarray()
        st.write(f"Dimensión de cada embedding: **{X.shape[1]}**")
        dims = st.radio("Proyección", ["2D", "3D"], horizontal=True)
        n = 3 if dims == "3D" and len(sents) >= 4 else 2
        P = PCA(n_components=n).fit_transform(X)
        df = pd.DataFrame(P, columns=["x", "y", "z"][:n])
        df["texto"] = sents
        fig = px.scatter_3d(df, x="x", y="y", z="z", text="texto") if n == 3 else px.scatter(df, x="x", y="y", text="texto")
        st.plotly_chart(fig, use_container_width=True)
        sim = cosine_similarity(X)
        labels = [s[:25] for s in sents]
        st.plotly_chart(px.imshow(sim, x=labels, y=labels, text_auto=".2f", color_continuous_scale="Viridis"), use_container_width=True)
        with st.expander("Ver vectores (primeras 20 dimensiones)"):
            st.dataframe(pd.DataFrame(X[:, :20].round(3), columns=vec.get_feature_names_out()[:20], index=labels))
    else:
        st.info("Ingresa al menos 3 frases.")


def generate(model, messages, temperature, top_p, max_tokens, seed=None):
    kwargs = dict(model=model, messages=messages, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
    if seed is not None:
        kwargs["seed"] = seed
    return client.chat.completions.create(**kwargs)


# ---------------- Generación ----------------
with tabs[4]:
    st.subheader("Generación de texto")
    c1, c2 = st.columns([1, 2])
    with c1:
        model = st.selectbox("Modelo", models)
        temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.05)
        top_p = st.slider("Top P", 0.0, 1.0, 1.0, 0.05)
        max_tokens = st.slider("Max tokens", 16, 4096, 512, 16)
        use_seed = st.checkbox("Usar seed")
        seed = st.number_input("Seed", value=42, step=1) if use_seed else None
        system = st.text_area("System prompt", "Eres un asistente útil que responde en español.")
    with c2:
        prompt = st.text_area("Prompt", "Explica qué es un token en un LLM en 3 oraciones.", height=150)
        if st.button("Generar", type="primary", disabled=client is None):
            with st.spinner("Generando..."):
                try:
                    r = generate(model, [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                                 temperature, top_p, max_tokens, int(seed) if seed is not None else None)
                    st.markdown(r.choices[0].message.content)
                    u = r.usage
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Tokens prompt", u.prompt_tokens)
                    m2.metric("Tokens respuesta", u.completion_tokens)
                    m3.metric("Total", u.total_tokens)
                    t = getattr(u, "total_time", None)
                    m4.metric("Tokens/seg", f"{u.completion_tokens / t:.0f}" if t else "—")
                except Exception as e:
                    st.error(f"Error: {e}")
        if client is None:
            st.warning("Ingresa tu API key de Groq en la barra lateral.")

# ---------------- Comparar ----------------
with tabs[5]:
    st.subheader("Comparar modelos y temperaturas")
    cmp_prompt = st.text_area("Prompt", "Escribe un eslogan creativo para una cafetería.", key="cmp")
    cmp_models = st.multiselect("Modelos", models, default=models[:2])
    temps = st.multiselect("Temperaturas", [0.0, 0.3, 0.7, 1.0, 1.5], default=[0.0, 1.0])
    cmp_max = st.slider("Max tokens", 16, 1024, 200, 16, key="cmp_max")
    if st.button("Comparar", disabled=client is None):
        rows = []
        for m in cmp_models:
            for t in temps:
                try:
                    r = generate(m, [{"role": "user", "content": cmp_prompt}], t, 1.0, cmp_max)
                    out = r.choices[0].message.content
                    rows.append({"Modelo": m, "Temperatura": t, "Tokens": r.usage.completion_tokens, "Respuesta": out})
                except Exception as e:
                    rows.append({"Modelo": m, "Temperatura": t, "Tokens": 0, "Respuesta": f"Error: {e}"})
        for row in rows:
            with st.container(border=True):
                st.markdown(f"**{row['Modelo']}** · T={row['Temperatura']} · {row['Tokens']} tokens")
                st.write(row["Respuesta"])
        texts = [r["Respuesta"] for r in rows]
        if len(texts) >= 2:
            X = TfidfVectorizer().fit_transform(texts)
            labels = [f"{r['Modelo'].split('/')[-1]} T={r['Temperatura']}" for r in rows]
            st.markdown("**Similitud coseno entre respuestas**")
            st.plotly_chart(px.imshow(cosine_similarity(X), x=labels, y=labels, text_auto=".2f"), use_container_width=True)
