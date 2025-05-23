import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
from ucimlrepo import fetch_ucirepo
import pickle
import os

def clean_feature_names(df):
    df.columns = [col.replace('[', '').replace(']', '') for col in df.columns]
    return df

def analysis_and_model_page():
    st.title("Анализ данных и модель предиктивного обслуживания")

    # Загрузка данных
    st.header("Загрузка данных")
    data_source = st.radio("Источник данных", ["UCI Repository", "CSV файл"])

    data = None

    if data_source == "UCI Repository":
        try:
            ai4i_2020_predictive_maintenance_dataset = fetch_ucirepo(id=601)
            data = pd.concat([ai4i_2020_predictive_maintenance_dataset.data.features,
                              ai4i_2020_predictive_maintenance_dataset.data.targets], axis=1)
            st.success("Данные успешно загружены из UCI Repository!")
        except Exception as e:
            st.error(f"Ошибка при загрузке данных: {e}")
    else:
        uploaded_file = st.file_uploader("Загрузите CSV файл", type="csv")
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                st.success("Данные успешно загружены из CSV файла!")
            except Exception as e:
                st.error(f"Ошибка при чтении файла: {e}")

    if data is not None:
        # Предобработка данных
        st.header("Предобработка данных")

        cols_to_drop = ['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
        cols_to_drop = [col for col in cols_to_drop if col in data.columns]
        data = data.drop(columns=cols_to_drop)

        if 'Type' in data.columns:
            le = LabelEncoder()
            data['Type'] = le.fit_transform(data['Type'])
            st.write("Категориальная переменная 'Type' преобразована в числовую.")

        # Очистка названий признаков
        data = clean_feature_names(data)

        # Обработка пропущенных значений
        st.subheader("Пропущенные значения")
        missing_counts = data.isnull().sum()
        st.write(missing_counts)

        if missing_counts.sum() > 0:
            strategy = st.selectbox("Выберите способ обработки пропущенных значений",
                                    ["Удалить строки с пропусками", "Заполнить средним", "Заполнить медианой"])

            if strategy == "Удалить строки с пропусками":
                data = data.dropna()
                st.write(f"Удалено строк: {missing_counts.sum()}")
            elif strategy == "Заполнить средним":
                data = data.fillna(data.mean())
                st.write("Пропуски заполнены средним значением")
            else:
                data = data.fillna(data.median())
                st.write("Пропуски заполнены медианой")

        # Разведочный анализ (EDA)
        st.header("Разведочный анализ данных (EDA)")
        st.subheader("Основные статистики")
        st.write(data.describe())

        st.subheader("Распределения признаков")
        num_cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
        selected_cols = st.multiselect("Выберите признаки для построения гистограмм", num_cols, default=num_cols[:5])
        if selected_cols:
            fig, axs = plt.subplots(len(selected_cols), 1, figsize=(6, 3 * len(selected_cols)))
            if len(selected_cols) == 1:
                axs = [axs]
            for ax, col in zip(axs, selected_cols):
                sns.histplot(data[col], bins=30, ax=ax, kde=True)
                ax.set_title(f"Распределение {col}")
            st.pyplot(fig)

        st.subheader("Матрица корреляций")
        corr = data.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)

        # Масштабирование числовых признаков
        numerical_features = ['Air temperature K', 'Process temperature K',
                              'Rotational speed rpm', 'Torque Nm', 'Tool wear min']
        numerical_features = [col for col in numerical_features if col in data.columns]
        scaler = None
        if numerical_features:
            scaler = StandardScaler()
            data[numerical_features] = scaler.fit_transform(data[numerical_features])
            st.write("Числовые признаки масштабированы.")

        # Разделение данных
        st.header("Разделение данных")
        X = data.drop(columns=['Machine failure'])
        y = data['Machine failure']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        st.write(f"Обучающая выборка: {X_train.shape[0]} записей")
        st.write(f"Тестовая выборка: {X_test.shape[0]} записей")

        # Обучение моделей
        st.header("Обучение моделей")
        model_option = st.selectbox("Выберите модель",
                                    ["Logistic Regression", "Random Forest", "XGBoost", "SVM"])

        kernel_type = None
        if model_option == "SVM":
            kernel_type = st.selectbox("Выберите ядро для SVM", ["linear", "rbf", "poly"], index=0)

        if st.button("Обучить модель"):
            try:
                if model_option == "Logistic Regression":
                    model = LogisticRegression(max_iter=1000)
                elif model_option == "Random Forest":
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                elif model_option == "XGBoost":
                    model = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
                else:
                    model = SVC(kernel=kernel_type, random_state=42, probability=True)

                model.fit(X_train, y_train)

                # Оценка модели
                st.header("Оценка модели")
                y_pred = model.predict(X_test)
                try:
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                except AttributeError:
                    y_pred_proba = model.decision_function(X_test)
                    y_pred_proba = (y_pred_proba - y_pred_proba.min()) / (y_pred_proba.max() - y_pred_proba.min())

                accuracy = accuracy_score(y_test, y_pred)
                st.write(f"Accuracy: {accuracy:.4f}")

                st.subheader("Confusion Matrix")
                fig, ax = plt.subplots()
                sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
                st.pyplot(fig)

                st.subheader("Classification Report")
                report = classification_report(y_test, y_pred, output_dict=True)
                st.table(pd.DataFrame(report).transpose())

                st.subheader("ROC Curve")
                if hasattr(model, "predict_proba") or hasattr(model, "decision_function"):
                    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                    roc_auc = roc_auc_score(y_test, y_pred_proba)
                    fig, ax = plt.subplots()
                    ax.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
                    ax.plot([0, 1], [0, 1], 'k--')
                    ax.set_xlabel('False Positive Rate')
                    ax.set_ylabel('True Positive Rate')
                    ax.set_title('ROC Curve')
                    ax.legend(loc="lower right")
                    st.pyplot(fig)
                else:
                    st.warning("Эта модель не поддерживает расчет вероятностей, ROC-кривая недоступна")

                # Сохранение модели
                st.session_state['model'] = model
                st.session_state['features'] = X.columns.tolist()
                st.session_state['scaler'] = scaler if numerical_features else None

                # Сохраняем модель на диск
                with open("saved_model.pkl", "wb") as f_out:
                    pickle.dump({
                        'model': model,
                        'features': X.columns.tolist(),
                        'scaler': scaler
                    }, f_out)
                st.success("Модель сохранена в saved_model.pkl")

            except Exception as e:
                st.error(f"Ошибка при обучении модели: {e}")

        # Загрузка модели с диска
        st.header("Загрузка ранее сохранённой модели")
        if os.path.exists("saved_model.pkl"):
            if st.button("Загрузить модель"):
                with open("saved_model.pkl", "rb") as f_in:
                    saved = pickle.load(f_in)
                    st.session_state['model'] = saved['model']
                    st.session_state['features'] = saved['features']
                    st.session_state['scaler'] = saved['scaler']
                st.success("Модель успешно загружена!")

        # Предсказание на новых данных (ручной ввод)
        st.header("Предсказание на новых данных (ручной ввод)")

        if 'model' in st.session_state and 'features' in st.session_state:
            with st.form("prediction_form"):
                st.write("Введите параметры оборудования:")
                input_data = {}
                if 'Type' in st.session_state['features']:
                    input_data['Type'] = st.selectbox("Тип оборудования", [0, 1, 2],
                                                      format_func=lambda x: ['L', 'M', 'H'][x])
                for feature in numerical_features:
                    input_data[feature] = st.number_input(f"{feature}", value=0.0)

                submitted = st.form_submit_button("Сделать предсказание")

                if submitted:
                    input_df = pd.DataFrame([input_data])

                    if st.session_state.get('scaler') is not None:
                        input_df[numerical_features] = st.session_state['scaler'].transform(input_df[numerical_features])

                    model = st.session_state['model']
                    prediction = model.predict(input_df)
                    try:
                        prediction_proba = model.predict_proba(input_df)[:, 1]
                    except AttributeError:
                        try:
                            decision = model.decision_function(input_df)
                            prediction_proba = (decision - decision.min()) / (decision.max() - decision.min())
                        except:
                            prediction_proba = [0.5]

                    st.subheader("Результат предсказания")
                    if prediction[0] == 1:
                        st.error(f"Прогнозируется отказ оборудования (вероятность: {prediction_proba[0]:.2%})")
                    else:
                        st.success(f"Оборудование в норме (вероятность отказа: {prediction_proba[0]:.2%})")



if __name__ == "__main__":
    analysis_and_model_page()