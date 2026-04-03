import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import json

st.set_page_config(
    page_title="Student Performance Analysis",
    page_icon="",
    layout="wide"
)

@st.cache_data
def load_data():
    return pd.read_csv("student_data.csv")

@st.cache_resource
def load_models():
    models = {
        "rf_reg_a": joblib.load("rf_reg_a.pkl"),
        "xgb_clf_a": joblib.load("xgb_clf_a.pkl"),
        "rf_reg_b": joblib.load("rf_reg_b.pkl"),
        "xgb_clf_b": joblib.load("xgb_clf_b.pkl"),
        "rf_clf_b": joblib.load("rf_clf_b.pkl"),
    }
    with open("model_b_columns.json", "r") as f:
        models["model_b_columns"] = json.load(f)
    return models

stu = load_data()
models = load_models()

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", [
    "Home",
    "Dataset Explorer",
    "EDA Visualisations",
    "Model Performance",
    "At-Risk Predictor"
])

if page == "Home":
    st.title("Predicting Student Academic Performance")
    st.markdown("**Eghonghon Aigbomian    |    20024813    |    Dublin Business School    |    B7IS138**")
    st.markdown("---")

    st.markdown("""
    This project investigates the factors influencing student academic performance 
    using the UCI Student Performance dataset (Cortez, 2008). The dataset includes 
    demographic, family background, behavioural, and academic attributes for students 
    across two subjects.
    """)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Students", "1,044")
    col2.metric("Variables", "34")
    col3.metric("Math Students", "395")
    col4.metric("Portuguese Students", "649")

    st.markdown("---")
    st.subheader("Two Modelling Strategies")

    col_a, col_b = st.columns(2)
    with col_a:
        st.info("""
        **Model A — Full Prediction**
        
        Uses all available features including prior period grades (G1 and G2).
        Answers: *What will this student's final grade be?*
        """)
    with col_b:
        st.warning("""
        **Model B — Early Warning**
        
        Removes G1 and G2 to simulate start-of-term prediction.
        Answers: *Is this student at risk of failing before any grades are in?*
        """)

    st.markdown("---")
    st.subheader("Key Findings at a Glance")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.success("G2 alone accounts for 82.5% of Model A feature importance (r = 0.911)")
    with col2:
        st.error("Students with 3 prior failures average G3 = 6.80 vs 12.05 for those with none")
    with col3:
        st.info("Model A best: RF Regression R² = 0.827 | Model B best: RF Regression R² = 0.239")


elif page == "Dataset Explorer":
    st.title("Dataset Explorer")
    st.markdown("Browse and filter the student dataset.")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        subject_filter = st.selectbox("Filter by Subject:", ["All", "math", "portuguese"])
    with col2:
        sex_filter = st.selectbox("Filter by Sex:", ["All", "F", "M"])

    filtered = stu.copy()
    if subject_filter != "All":
        filtered = filtered[filtered["subject"] == subject_filter]
    if sex_filter != "All":
        filtered = filtered[filtered["sex"] == sex_filter]

    st.write(f"Showing **{len(filtered)}** students")

    display_cols = ["school", "sex", "age", "address", "Medu", "Fedu", "Mjob", "Fjob",
                    "studytime", "failures", "absences", "higher", "Walc", "Dalc",
                    "G1", "G2", "G3", "subject"]
    st.dataframe(filtered[display_cols].reset_index(drop=True), use_container_width=True)

    st.markdown("---")
    st.subheader("Summary Statistics")
    st.dataframe(filtered[["age", "studytime", "failures", "absences", "G1", "G2", "G3"]].describe().round(2), use_container_width=True)

    st.markdown("---")
    st.subheader("Grade Distribution")
    fig = px.histogram(filtered, x="G3", nbins=20,
                       title=f"Distribution of Final Grades (G3) — {subject_filter}",
                       labels={"G3": "Final Grade (G3)", "count": "Frequency"},
                       color_discrete_sequence=["#2C5F8A"])
    fig.update_layout(bargap=0.1)
    st.plotly_chart(fig, use_container_width=True)


elif page == "EDA Visualisations":
    st.title("Exploratory Data Analysis")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["Variable vs G3", "Correlation Heatmap", "Pairwise Comparisons"])

    with tab1:
        st.subheader("Explore Any Variable Against Final Grade (G3)")

        variable = st.selectbox("Choose a variable to plot against G3:", [
            "failures", "studytime", "Walc", "Dalc", "goout",
            "freetime", "Medu", "Fedu", "higher", "subject",
            "sex", "address", "Mjob", "Fjob", "health"
        ])

        group_means = stu.groupby(variable)["G3"].mean().reset_index()
        group_means.columns = [variable, "Mean G3"]
        group_means["Mean G3"] = group_means["Mean G3"].round(2)

        fig_box = px.box(stu, x=variable, y="G3",
                         title=f"G3 vs {variable}",
                         labels={"G3": "Final Grade (G3)", variable: variable},
                         color=variable)
        st.plotly_chart(fig_box, use_container_width=True)

        st.write("Group Mean G3:")
        st.dataframe(group_means, use_container_width=True)

        interpretations = {
            "failures": "Clear negative trend. Mean G3 drops from 12.05 (0 failures) to 6.80 (3 failures). Correlation: -0.383. The strongest non-grade predictor.",
            "studytime": "Weak positive trend. Mean G3 rises from 10.58 (category 1) to 12.49 (category 3) before dipping slightly at category 4.",
            "Walc": "Weak negative relationship. Higher weekend alcohol consumption is associated with slightly lower grades but is not a dominant predictor.",
            "Dalc": "Similar to Walc. Weak negative relationship. Dalc and Walc are strongly correlated with each other.",
            "goout": "Very weak negative trend. Going out frequency has minimal impact on grades.",
            "freetime": "No consistent trend. Grades fluctuate without a clear direction across free time levels.",
            "Medu": "Moderate positive relationship. Students with more educated mothers tend to score higher. Correlation with G3: 0.201.",
            "Fedu": "Similar to Medu but slightly weaker. Correlation with G3: 0.160.",
            "higher": "Strong difference. Students planning for higher education average 11.62 vs 8.35 for those who do not.",
            "subject": "Moderate difference. Portuguese students (mean 11.91) outperform Math students (mean 10.42).",
            "sex": "Very weak. Females average 11.45, males 11.20 — a negligible difference of 0.25 grade points.",
            "address": "Weak. Urban students show a marginally higher median than rural students.",
            "Mjob": "Moderate. Children of mothers in health and teacher roles tend to perform best.",
            "Fjob": "Similar to Mjob. Father's occupation as teacher shows the highest mean G3.",
            "health": "Very weak. Self-reported health has minimal relationship with final grade."
        }

        if variable in interpretations:
            st.info(f"**Interpretation:** {interpretations[variable]}")

    with tab2:
        st.subheader("Correlation Heatmap — All Numeric Variables")

        num_cols = stu.select_dtypes(include=["int64", "float64"]).columns.tolist()
        corr = stu[num_cols].corr().round(2)

        fig_heat = px.imshow(
            corr,
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            title="Correlation Matrix",
            text_auto=True,
            aspect="auto"
        )
        fig_heat.update_layout(height=700)
        st.plotly_chart(fig_heat, use_container_width=True)

        st.markdown("---")
        st.subheader("Strongest Correlations with G3")
        g3_corr = stu[num_cols].corr()["G3"].drop("G3").sort_values(ascending=False).reset_index()
        g3_corr.columns = ["Variable", "Correlation with G3"]
        g3_corr["Correlation with G3"] = g3_corr["Correlation with G3"].round(3)

        fig_bar = px.bar(
            g3_corr, x="Variable", y="Correlation with G3",
            color="Correlation with G3",
            color_continuous_scale="RdBu_r",
            title="Correlation of Each Variable with G3",
            range_color=[-1, 1]
        )
        fig_bar.add_hline(y=0, line_dash="dash", line_color="black")
        st.plotly_chart(fig_bar, use_container_width=True)

    with tab3:
        st.subheader("G3 vs Early Assessment Scores")

        grade_var = st.selectbox("Compare G3 against:", ["G1", "G2"])
        fig_scatter = px.scatter(
            stu, x=grade_var, y="G3",
            color="subject",
            opacity=0.5,
            trendline="ols",
            title=f"G3 vs {grade_var} with Regression Line",
            labels={grade_var: f"{grade_var} (Period Grade)", "G3": "Final Grade (G3)"}
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        corr_val = stu[grade_var].corr(stu["G3"])
        st.info(f"**Correlation between {grade_var} and G3: {corr_val:.3f}** — {'Very strong positive relationship' if corr_val > 0.8 else 'Strong positive relationship'}")

        st.markdown("---")
        st.subheader("Subject Comparison by Study Time")
        fig_hue = px.box(stu, x="studytime", y="G3", color="subject",
                         title="G3 vs Study Time by Subject",
                         labels={"studytime": "Weekly Study Time", "G3": "Final Grade (G3)"})
        st.plotly_chart(fig_hue, use_container_width=True)


elif page == "Model Performance":
    st.title("Model Performance Comparison")
    st.markdown("---")

    tab1, tab2 = st.tabs(["Regression Models", "Classification Models"])

    with tab1:
        st.subheader("Regression Results")

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("**Model A — With Prior Grades (G1, G2)**")
            reg_a = pd.DataFrame({
                "Model": ["Linear A", "Ridge A", "Lasso A", "RF A", "XGB A"],
                "R²": [0.800, 0.800, 0.807, 0.827, 0.823],
                "RMSE": [1.760, 1.758, 1.726, 1.636, 1.654]
            })
            st.dataframe(reg_a.set_index("Model"), use_container_width=True)

        with col_b:
            st.markdown("**Model B — Without Prior Grades (Early Warning)**")
            reg_b = pd.DataFrame({
                "Model": ["Linear B", "Ridge B", "Lasso B", "RF B", "XGB B"],
                "R²": [0.119, 0.121, 0.136, 0.239, 0.106],
                "RMSE": [3.692, 3.686, 3.654, 3.431, 3.719]
            })
            st.dataframe(reg_b.set_index("Model"), use_container_width=True)

        st.markdown("---")
        st.subheader("R² Score Comparison — All Models")

        all_reg = pd.DataFrame({
            "Model": ["Linear A", "Ridge A", "Lasso A", "RF A", "XGB A",
                      "Linear B", "Ridge B", "Lasso B", "RF B", "XGB B"],
            "R²": [0.800, 0.800, 0.807, 0.827, 0.823,
                   0.119, 0.121, 0.136, 0.239, 0.106],
            "Set": ["Model A", "Model A", "Model A", "Model A", "Model A",
                    "Model B", "Model B", "Model B", "Model B", "Model B"]
        })

        fig_r2 = px.bar(all_reg, x="Model", y="R²", color="Set",
                        barmode="group",
                        title="R² Scores Across All Regression Models",
                        color_discrete_map={"Model A": "#2C5F8A", "Model B": "#E07B39"},
                        text="R²")
        fig_r2.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig_r2.update_layout(yaxis_range=[0, 1])
        st.plotly_chart(fig_r2, use_container_width=True)

        st.info("""
        **Key insight:** The dramatic drop in R² from Model A (0.80-0.83) to Model B (0.11-0.24) 
        confirms that prior grades (G1, G2) are the dominant predictors. Without them, predicting 
        exact final grades is significantly harder. Random Forest consistently outperforms linear 
        models in Model B, detecting non-linear signals that correlation cannot capture.
        """)
    
        st.markdown("---")
        st.subheader("Actual vs Predicted — Random Forest Regression")

        model_choice = st.selectbox("Select model set:", ["Model A (RF)", "Model B (RF)"])

        from sklearn.model_selection import train_test_split
        stu_model = stu.copy()
        stu_model["at_risk"] = (stu_model["G3"] < 10).astype(int)
        y_reg = stu_model["G3"]

        X_full = stu_model.drop(["G3", "at_risk"], axis=1)
        X_full = pd.get_dummies(X_full, drop_first=True)
        X_no_grades = stu_model.drop(["G3", "G1", "G2", "at_risk"], axis=1)
        X_no_grades = pd.get_dummies(X_no_grades, drop_first=True)

        if model_choice == "Model A (RF)":
            Xf_train, Xf_test, yf_train, yf_test = train_test_split(X_full, y_reg, test_size=0.2, random_state=42)
            y_pred = models["rf_reg_a"].predict(Xf_test)
            y_true = yf_test
            title = "RF Regression Model A — Actual vs Predicted"
        else:
            Xn_train, Xn_test, yn_train, yn_test = train_test_split(X_no_grades, y_reg, test_size=0.2, random_state=42)
            y_pred = models["rf_reg_b"].predict(Xn_test)
            y_true = yn_test
            title = "RF Regression Model B — Actual vs Predicted"

        fig_avp = px.scatter(
            x=y_true, y=y_pred,
            opacity=0.6,
            labels={"x": "Actual G3", "y": "Predicted G3"},
            title=title
        )
        fig_avp.add_shape(type="line", x0=0, y0=0, x1=20, y1=20,
                          line=dict(color="red", dash="dash"), name="Perfect Prediction")
        st.plotly_chart(fig_avp, use_container_width=True)

    with tab2:
        st.subheader("Classification Results")

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("**Model A — With Prior Grades**")
            clf_a = pd.DataFrame({
                "Model": ["Logistic Reg A", "RF A", "XGB A"],
                "Accuracy": [0.885, 0.885, 0.895],
                "Precision": [0.860, 0.891, 0.880],
                "Recall": [0.717, 0.683, 0.733],
                "F1": [0.782, 0.774, 0.800]
            })
            st.dataframe(clf_a.set_index("Model"), use_container_width=True)

        with col_b:
            st.markdown("**Model B — Without Prior Grades**")
            clf_b = pd.DataFrame({
                "Model": ["Logistic Reg B", "RF B", "XGB B"],
                "Accuracy": [0.737, 0.742, 0.732],
                "Precision": [0.609, 0.625, 0.553],
                "Recall": [0.233, 0.250, 0.350],
                "F1": [0.337, 0.357, 0.429]
            })
            st.dataframe(clf_b.set_index("Model"), use_container_width=True)

        st.markdown("---")
        st.subheader("Metric Comparison — Classification Models")

        metric_choice = st.selectbox("Select metric to compare:", ["Accuracy", "Precision", "Recall", "F1"])

        metric_data = pd.DataFrame({
            "Model": ["LogReg A", "RF A", "XGB A", "LogReg B", "RF B", "XGB B"],
            "Accuracy": [0.885, 0.885, 0.895, 0.737, 0.742, 0.732],
            "Precision": [0.860, 0.891, 0.880, 0.609, 0.625, 0.553],
            "Recall": [0.717, 0.683, 0.733, 0.233, 0.250, 0.350],
            "F1": [0.782, 0.774, 0.800, 0.337, 0.357, 0.429],
            "Set": ["Model A", "Model A", "Model A", "Model B", "Model B", "Model B"]
        })

        fig_clf = px.bar(metric_data, x="Model", y=metric_choice, color="Set",
                         title=f"{metric_choice} Across All Classification Models",
                         color_discrete_map={"Model A": "#2C5F8A", "Model B": "#E07B39"},
                         text=metric_choice)
        fig_clf.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig_clf.update_layout(yaxis_range=[0, 1.1])
        st.plotly_chart(fig_clf, use_container_width=True)

        st.markdown("---")
        st.subheader("Confusion Matrix — XGBoost Classifier")

        cm_choice = st.selectbox("Select model:", ["Model A", "Model B"])

        from sklearn.model_selection import train_test_split
        stu_cm = stu.copy()
        stu_cm["at_risk"] = (stu_cm["G3"] < 10).astype(int)
        y_clf = stu_cm["at_risk"]

        X_full_cm = pd.get_dummies(stu_cm.drop(["G3", "at_risk"], axis=1), drop_first=True)
        X_no_grades_cm = pd.get_dummies(stu_cm.drop(["G3", "G1", "G2", "at_risk"], axis=1), drop_first=True)

        if cm_choice == "Model A":
            Xf_c_tr, Xf_c_te, yf_c_tr, yf_c_te = train_test_split(X_full_cm, y_clf, test_size=0.2, random_state=42)
            y_pred_cm = models["xgb_clf_a"].predict(Xf_c_te)
            y_true_cm = yf_c_te
        else:
            Xn_c_tr, Xn_c_te, yn_c_tr, yn_c_te = train_test_split(X_no_grades_cm, y_clf, test_size=0.2, random_state=42)
            y_pred_cm = models["xgb_clf_b"].predict(Xn_c_te)
            y_true_cm = yn_c_te

        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true_cm, y_pred_cm)
        labels = ["Not At Risk", "At Risk"]
        fig_cm = px.imshow(
            cm,
            x=labels, y=labels,
            text_auto=True,
            color_continuous_scale="Blues",
            title=f"Confusion Matrix — XGBoost {cm_choice}",
            labels=dict(x="Predicted", y="Actual")
        )
        fig_cm.update_layout(height=400)
        st.plotly_chart(fig_cm, use_container_width=True)

        if cm_choice == "Model A":
            st.success("Model A correctly identifies 73% of at-risk students (Recall = 0.733). Only 16 of 60 at-risk students were missed.")
        else:
            st.warning("Model B misses 65% of at-risk students (Recall = 0.350). Without prior grades, early warning is significantly harder but failures and absences provide a useful starting signal.")


elif page == "At-Risk Predictor":
    st.title("Early Warning: At-Risk Student Predictor")
    st.markdown("""
    Enter a student's details below. The model will predict whether they are likely 
    to score below 10 (at risk of failing) using **only demographic and behavioural 
    information — no prior grades required**. This uses the trained Model B XGBoost classifier.
    """)
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Academic History")
        failures = st.slider("Number of Prior Failures", 0, 3, 0)
        studytime = st.select_slider(
            "Weekly Study Time",
            options=[1, 2, 3, 4],
            value=2,
            format_func=lambda x: {1: "< 2 hours", 2: "2-5 hours", 3: "5-10 hours", 4: "> 10 hours"}[x]
        )
        absences = st.slider("Number of Absences", 0, 75, 4)
        schoolsup = st.selectbox("Receives Extra School Support?", ["no", "yes"])
        higher = st.selectbox("Plans for Higher Education?", ["yes", "no"])

    with col2:
        st.subheader("Family Background")
        Medu = st.select_slider(
            "Mother's Education Level",
            options=[0, 1, 2, 3, 4],
            value=2,
            format_func=lambda x: {0: "None", 1: "Primary", 2: "Middle School", 3: "Secondary", 4: "Higher Education"}[x]
        )
        Fedu = st.select_slider(
            "Father's Education Level",
            options=[0, 1, 2, 3, 4],
            value=2,
            format_func=lambda x: {0: "None", 1: "Primary", 2: "Middle School", 3: "Secondary", 4: "Higher Education"}[x]
        )
        Mjob = st.selectbox("Mother's Job", ["at_home", "health", "other", "services", "teacher"])
        Fjob = st.selectbox("Father's Job", ["at_home", "health", "other", "services", "teacher"])
        famrel = st.slider("Family Relationship Quality (1=very bad, 5=excellent)", 1, 5, 4)

    with col3:
        st.subheader("Lifestyle")
        goout = st.slider("Going Out with Friends (1=very low, 5=very high)", 1, 5, 2)
        Walc = st.slider("Weekend Alcohol Consumption (1=very low, 5=very high)", 1, 5, 1)
        Dalc = st.slider("Weekday Alcohol Consumption (1=very low, 5=very high)", 1, 5, 1)
        freetime = st.slider("Free Time After School (1=very low, 5=very high)", 1, 5, 3)
        health = st.slider("Health Status (1=very bad, 5=very good)", 1, 5, 3)
        subject = st.selectbox("Subject", ["math", "portuguese"])
        sex = st.selectbox("Sex", ["F", "M"])
        address = st.selectbox("Address Type", ["U", "R"])
        age = st.slider("Age", 15, 22, 17)
        school = st.selectbox("School", ["GP", "MS"])

    st.markdown("---")

    if st.button("Predict At-Risk Status", type="primary"):
        input_dict = {
            "school": school,
            "sex": sex,
            "age": age,
            "address": address,
            "famsize": "GT3",
            "Pstatus": "T",
            "Medu": Medu,
            "Fedu": Fedu,
            "Mjob": Mjob,
            "Fjob": Fjob,
            "reason": "course",
            "guardian": "mother",
            "traveltime": 1,
            "studytime": studytime,
            "failures": failures,
            "schoolsup": schoolsup,
            "famsup": "yes",
            "paid": "no",
            "activities": "no",
            "nursery": "yes",
            "higher": higher,
            "internet": "yes",
            "romantic": "no",
            "famrel": famrel,
            "freetime": freetime,
            "goout": goout,
            "Dalc": Dalc,
            "Walc": Walc,
            "health": health,
            "absences": absences,
            "subject": subject
        }

        input_df = pd.DataFrame([input_dict])
        input_dummies = pd.get_dummies(input_df, drop_first=True)

        model_b_columns = models["model_b_columns"]
        for col in model_b_columns:
            if col not in input_dummies.columns:
                input_dummies[col] = 0
        input_dummies = input_dummies[model_b_columns]

        prediction = models["xgb_clf_b"].predict(input_dummies)[0]
        probability = models["xgb_clf_b"].predict_proba(input_dummies)[0]

        col_result, col_prob = st.columns(2)

        with col_result:
            if prediction == 1:
                st.error("⚠️ AT RISK — This student is predicted to score below 10 on their final grade.")
            else:
                st.success("✅ NOT AT RISK — This student is predicted to pass their final assessment.")

        with col_prob:
            prob_df = pd.DataFrame({
                "Outcome": ["Not At Risk", "At Risk"],
                "Probability": [round(probability[0], 3), round(probability[1], 3)]
            })
            fig_prob = px.bar(prob_df, x="Outcome", y="Probability",
                              color="Outcome",
                              color_discrete_map={"Not At Risk": "#28a745", "At Risk": "#dc3545"},
                              title="Prediction Confidence",
                              text="Probability")
            fig_prob.update_traces(texttemplate="%{text:.1%}", textposition="outside")
            fig_prob.update_layout(yaxis_range=[0, 1.2], showlegend=False)
            st.plotly_chart(fig_prob, use_container_width=True)

        st.caption("This prediction uses demographic and behavioural data only, with no prior grade history. It is a decision-support tool, not a definitive assessment. Model B recall = 0.350, meaning it misses approximately 65% of truly at-risk students.")
