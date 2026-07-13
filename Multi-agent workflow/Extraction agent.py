from typing import Dict, Optional

import ast
import json
import os
import re
from pathlib import Path
from dotenv import load_dotenv
import numpy as np

load_dotenv(Path(__file__).resolve().parent / ".env")
import pandas as pd
import shap
import xgboost as xgb
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from pandas.api.types import is_object_dtype, is_string_dtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_openai_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set in environment variables or .env file")
    return api_key


OPENAI_API_KEY = load_openai_api_key()


def process_data(df, cat_cols, num_cols, id_feats, target):
    missing_cols = set(cat_cols + num_cols + [target]) - set(df.columns)
    if missing_cols:
        raise KeyError(f"Missing columns: {sorted(missing_cols)}")

    df[target] = df[target].apply(lambda x: 1 if x == 'Yes' else 0)
    # Convert categorical columns to 'category' dtype
    df[cat_cols] = df[cat_cols].astype('category')

    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['candidateid'] = df['candidateid'].astype('category')

    return df
def generate_text_summary(df, num_feats, cat_feats):
    summary = []
    summary.append(f"Dataset Overview: Contains {df.shape[0]} rows and {df.shape[1]} columns.")
    missing_data = df.isnull().sum().sum()
    summary.append(f"Missing Values: Total missing entries {missing_data}.")

    # Handling numeric features
    for column in num_feats:
        summary.append(f"\n{column}: distribution across different percentiles")
        if df[column].dtype == 'float64' or df[column].dtype == 'int64':
            stats = df[column].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
            summary.append(f"Mean: {stats['mean']:.2f}, Std: {stats['std']:.2f}")
            summary.append(f"Min: {stats['min']:.2f}, 10th Percentile: {stats['10%']:.2f}, "
                           f"25th Percentile: {stats['25%']:.2f}, Median (50th Percentile): {stats['50%']:.2f}, "
                           f"75th Percentile: {stats['75%']:.2f}, 90th Percentile: {stats['90%']:.2f}, Max: {stats['max']:.2f}")
            summary.append(f"Missing: {df[column].isnull().sum()} ({df[column].isnull().mean() * 100:.2f}%)")

    # Handling categorical features
    for column in cat_feats:
        summary.append(f"\n{column}: distribution across top 10 categories")
        if isinstance(df[column].dtype, pd.CategoricalDtype):
            top_categories = df[column].value_counts().nlargest(10).to_dict()
            summary.append(f"Top Categories distribution: {top_categories}")
            summary.append(f"Missing values: {df[column].isnull().sum()} ({df[column].isnull().mean() * 100:.2f}%)")

    # Analyzing correlations among numeric features
    if num_feats:
        correlations = df[num_feats].corr()
        summary.append("\nPearson correlation among numeric features:")
        highly_correlated_pairs = correlations.where(np.triu(np.abs(correlations) > 0.5, k=1)).stack()
        for (feature1, feature2), corr_value in highly_correlated_pairs.items():
            summary.append(f"\n{feature1} and {feature2}: have high correlation of {corr_value:.2f}")
    op = " ".join(summary)

    return " ".join(summary)



class MultimodalShapAnalyzer:
    def __init__(self, model, X_train, dtrain, cat_features, num_features,
                 captions_df: Optional[pd.DataFrame] = None,
                 image_col: str = 'picindex',
                 caption_col: str = 'caption'):
        self.model = model
        self.X_train = X_train
        self.dtrain = dtrain
        self.cat_features = cat_features
        self.num_features = num_features
        self.captions_df = captions_df
        self.image_col = image_col
        self.caption_col = caption_col
        def _basename_lower(p):
            s = str(p).strip().strip('"').strip("'").replace("\\", "/")
            return os.path.basename(s).lower()

        self._basename_lower = _basename_lower

        self._caption_map = {}
        if self.captions_df is not None:
            if (self.image_col in self.captions_df.columns) and (self.caption_col in self.captions_df.columns):
                for _, r in self.captions_df.iterrows():
                    key = _basename_lower(r[self.image_col])
                    cap = str(r[self.caption_col]).strip() if pd.notna(r[self.caption_col]) else ""
                    if key and cap:
                        self._caption_map[key] = cap
                        if "." not in key:
                            self._caption_map[key + ".jpg"] = cap
                            self._caption_map[key + ".png"] = cap

        self.explainer = shap.TreeExplainer(model)
        self.shap_values = self.explainer.shap_values(dtrain)
        self.base_value = self.explainer.expected_value

        self.result_df = None
        self.importance_df = None
        self.summary_df = None

    def _get_caption(self, image_path: str) -> str:
        if not self._caption_map:
            return ""
        key = self._basename_lower(image_path)
        return self._caption_map.get(key, "")


    def get_shap_value(self):
        return self.shap_values

    def analyze_shap_values(self, include_captions: bool = False):

        results = []
        feature_importances = {}


        for feature in self.cat_features + self.num_features:
            feature_shap_values = self.shap_values[:, self.X_train.columns.get_loc(feature)]
            feature_importances[feature] = np.mean(np.abs(feature_shap_values))


        importance_df = pd.DataFrame(list(feature_importances.items()),
                                     columns=['Feature', 'Importance'])
        importance_df.sort_values('Importance', ascending=False, inplace=True)
        importance_df['Rank'] = range(1, len(importance_df) + 1)
        importance_ranks = importance_df.set_index('Feature')['Rank'].to_dict()


        for feature in self.cat_features + self.num_features:
            feature_values = self.X_train[feature]
            feature_shap_values = self.shap_values[:, self.X_train.columns.get_loc(feature)]
            df = pd.DataFrame({feature: feature_values, 'SHAP Value': feature_shap_values})


            if include_captions and self.image_col in self.X_train.columns:
                df['Caption'] = self.X_train[self.image_col].apply(self._get_caption)


            if feature in self.num_features:
                df['Group'] = pd.qcut(df[feature], 10, duplicates='drop')
            else:
                df['Group'] = df[feature]


            group_cols = ['Group', 'Caption'] if include_captions and 'Caption' in df.columns else ['Group']
            group_avg = df.groupby(group_cols, observed=True)['SHAP Value'].mean().reset_index()

            group_avg['Adjusted Value'] = self.base_value + group_avg['SHAP Value']
            group_avg['Change in Value'] = group_avg['Adjusted Value'] - self.base_value
            group_avg['Feature'] = feature
            group_avg['Feature Importance'] = feature_importances[feature]
            group_avg['Importance Rank'] = importance_ranks[feature]

            results.append(group_avg)

        self.result_df = pd.concat(results, ignore_index=True)
        self.importance_df = importance_df
        return self.result_df

    def summarize_shap_text(self, include_visual: bool = False):

        descriptions = []
        descriptions.append("""Below is the description of partial dependence (PD) of target prediction on all the features.
They help in understanding how the features affect the predictions of a model, regardless of the values of other features.\n""")

        if include_visual and 'Caption' in self.result_df.columns:
            descriptions.append(
                "\nNOTE: The following analysis also considers visual descriptions from experimental images.\n")

        for _, row in self.result_df.iterrows():
            feature = row['Feature']
            effect = "increases" if row['Change in Value'] > 0 else "decreases"
            change = abs(row['Change in Value'])

            if isinstance(row['Group'], pd.Interval):
                desc = f"When {feature} is within {row['Group']}, it {effect} the predicted value by {change:.2f}."
            else:
                desc = f"When {feature} is {row['Group']}, it {effect} the predicted value by {change:.2f}."

            if include_visual and 'Caption' in row and pd.notna(row['Caption']):
                desc += f" Related image shows: {row['Caption']}"

            descriptions.append(desc)

        importance_text = [
            "\n\nFeature importance from the SHAP summary:",
            "The mean absolute SHAP value provides an aggregate measure of the overall impact that each feature has on the model's predictions."
        ]

        if include_visual and self.captions_df is not None:
            importance_text.append("\nNOTE: Visual features from images were also considered in this analysis.")

        imp_df = self.importance_df[['Feature', 'Importance', 'Rank']].drop_duplicates()
        imp_df.sort_values('Rank', ascending=True, inplace=True)

        for _, row in imp_df.iterrows():
            importance_text.append(
                f"The importance rank of {row['Feature']} is {row['Rank']} (SHAP value: {row['Importance']:.4f}).")

        return "\n".join(descriptions + importance_text)

    def summarize_shap_df(self):
        results = []

        for _, row in self.result_df.iterrows():
            feature = row['Feature']
            effect = "increases" if row['Change in Value'] > 0 else "decreases"
            change = abs(row['Change in Value'])

            group = str(row['Group']) if isinstance(row['Group'], pd.Interval) else row['Group']

            result = {
                'feature': feature,
                'feature_group': group,
                'feature_effect': effect,
                'value_contribution': change,
                'Feature_Importance_Rank': row['Importance Rank'],
                'SHAP_Value': row['SHAP Value']
            }

            if 'Caption' in row:
                result['image_caption'] = row['Caption']

            results.append(result)

        self.summary_df = pd.DataFrame(results)
        return self.summary_df

    def extract_shap_trend_summary(self, include_visual: bool = False):

        summary = {}

        for feature in self.result_df['Feature'].unique():
            sub_df = self.result_df[self.result_df['Feature'] == feature].copy()

            try:
                sub_df = sub_df.sort_values(by='Group')
            except:
                pass

            trend_lines = []
            for _, row in sub_df.iterrows():
                group = row['Group']
                shap_val = row['SHAP Value']
                if pd.isnull(shap_val):
                    continue


                group_str = f"[{group.left:.2f}, {group.right:.2f})" if isinstance(group, pd.Interval) else str(group)


                if shap_val > 0.2:
                    effect = "Strong Positive"
                elif shap_val > 0:
                    effect = "Positive"
                elif shap_val < -0.2:
                    effect = "Strong Negative"
                elif shap_val < 0:
                    effect = "Negative"
                else:
                    effect = "Neutral"

                line = f"- Group {group_str}: SHAP {shap_val:+.2f} 閳?{effect}"

                if include_visual and 'Caption' in row and pd.notna(row['Caption']):
                    line += f" (Image shows: {row['Caption']})"

                trend_lines.append(line)

            summary_text = f"For feature '{feature}':\n" + "\n".join(trend_lines)
            summary[feature] = summary_text

        return summary


def _run_extraction(data_csv_path, caption_csv_path):
    """Core extraction logic, parameterized by input paths."""
    local_file = data_csv_path
    data = pd.read_csv(local_file)



    X = data[
        [
            'surfactant_in_water',
            'oil_type',
            'surfactant_in_oil',
            'ratio_of_surfactant_in_water',
            'ratio_of_surfactant_in_oil',
            'ion_concentration_in_water',
        ]
    ].copy()

    y = pd.to_numeric(data['reward'], errors='raise')

    label_encoders = {}
    encoder_mappings = {}

    for column in X.columns:
        if is_object_dtype(X[column]) or is_string_dtype(X[column]):
            le = LabelEncoder()

            if X[column].isna().any():
                raise ValueError(f"Categorical column {column!r} contains missing values")

            X[column] = le.fit_transform(X[column].astype(str)).astype("int64")
            label_encoders[column] = le
            encoder_mappings[column] = {
                label: int(code)
                for label, code in zip(le.classes_, le.transform(le.classes_))
            }

    X = X.apply(pd.to_numeric, errors="raise").astype("float64")


    invalid_columns = X.select_dtypes(exclude=["number", "bool"]).columns.tolist()
    if invalid_columns:
        raise TypeError(f"Non-numeric model features remain: {invalid_columns}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)


    def merge_image_descriptors_csv(
        desc_csv_path: str,
        csv_input_path: str,
        image_col_in_input: str = "picindex",
        image_col_in_desc: str = "image",
        make_caption: bool = True,
        encoding_input: str = "utf-8",
        encoding_desc: str = "utf-8",
    ):



        df_in = pd.read_csv(csv_input_path, encoding=encoding_input)
        if image_col_in_input not in df_in.columns:
            raise KeyError(f"Input CSV is missing {image_col_in_input!r}; columns: {list(df_in.columns)}")


        df_desc = pd.read_csv(desc_csv_path, sep=None, engine="python", encoding=encoding_desc)
        if image_col_in_desc not in df_desc.columns:
            raise KeyError(f"Descriptor CSV is missing {image_col_in_desc!r}; columns: {list(df_desc.columns)}")


        def _basename_clean(x: str) -> str:
            s = str(x)

            s = s.strip().strip('"').strip("'").replace("\\", "/")
            return os.path.basename(s).lower()

        df_in["_img_key"] = df_in[image_col_in_input].map(_basename_clean)
        df_desc["_img_key"] = df_desc[image_col_in_desc].map(_basename_clean)


        if make_caption:
            def _mk_caption(r):
                parts = []
                if "phase" in r and pd.notna(r["phase"]): parts.append(f"phase={r['phase']}")
                if "cluster_uniformity" in r and pd.notna(r["cluster_uniformity"]): parts.append(f"uniformity={r['cluster_uniformity']}")
                if "residue_level" in r and pd.notna(r["residue_level"]): parts.append(f"residue={r['residue_level']}")
                if "interface_cross" in r and pd.notna(r["interface_cross"]): parts.append(f"interface_cross={r['interface_cross']}")
                if "satellite_cluster_ratio" in r and pd.notna(r["satellite_cluster_ratio"]): parts.append(f"satellite={r['satellite_cluster_ratio']}")
                return "; ".join(parts) if parts else ""
            df_desc["caption"] = df_desc.apply(_mk_caption, axis=1)


        keep = [c for c in df_desc.columns if c != image_col_in_desc]  # 閸樼粯甯€閸?image 閸掓绱濋柆鍨帳闁插秴顦?
        merged = df_in.merge(df_desc[keep], on="_img_key", how="left").drop(columns=["_img_key"])


        return merged


    data_with_captions = merge_image_descriptors_csv(
        desc_csv_path=caption_csv_path,
        csv_input_path=data_csv_path,
        image_col_in_input="picindex",
        image_col_in_desc="image",
        make_caption=True,
        encoding_desc="gbk",
    )


    # Category mappings are kept in memory for the RAG context.
    def to_group_mapping(mapping: dict) -> dict:
        return {k: f"group{v}" for k, v in mapping.items()}

    mapping_lines = ["Categorical variable label mappings (label-encoded to group names)"]
    for col, mapping in encoder_mappings.items():
        mapping_lines.append(f"{col}: {to_group_mapping(mapping)}")
    category_mapping_text = "\n".join(mapping_lines)


    dtrain = xgb.DMatrix(X, label=y)

    X_train_with_pic = X.copy()
    X_train_with_pic['picindex'] = data.loc[X.index, 'picindex']



    cat_features = ['surfactant_in_water', 'oil_type', 'surfactant_in_oil']

    num_features = [col for col in X.columns if col not in cat_features]




    shap_analyzer = MultimodalShapAnalyzer(
        model=model,
        X_train=X_train_with_pic,
        dtrain=dtrain,
        cat_features=cat_features,
        num_features=num_features,
        captions_df=data_with_captions,
        image_col='picindex',
        caption_col='caption'
    )



    shap_analyzer.analyze_shap_values(include_captions=True)


    shap_text_summary = shap_analyzer.summarize_shap_text(include_visual=True)

    # Generate multimodal SHAP summaries in memory.
    shap_trend_summary = shap_analyzer.extract_shap_trend_summary(include_visual=True)
    trend_text = "\n\n".join(shap_trend_summary.values())


    def apply_group_mapping(mapping_text: str, input_text: str) -> str:
        mappings: Dict[str, Dict[int, str]] = {}
        for line in mapping_text.splitlines():
            if ":" not in line or "{" not in line or "}" not in line:
                continue
            key, dict_str = line.split(":", 1)
            try:
                raw = ast.literal_eval(dict_str.strip())
            except (SyntaxError, ValueError):
                continue
            inverse = {}
            for name, group in raw.items():
                match = re.search(r"group\s*(\d+)", str(group), flags=re.IGNORECASE)
                if match:
                    inverse[int(match.group(1))] = str(name)
            if inverse:
                mappings[key.strip()] = inverse

        output = input_text
        for feature, group_to_name in mappings.items():
            block_pattern = re.compile(
                rf"(For feature\s*'{re.escape(feature)}'\s*:\s*)(.*?)(?=(?:\nFor feature\s*')|\Z)",
                flags=re.IGNORECASE | re.DOTALL,
            )

            def replace_block(block_match):
                def replace_group(group_match):
                    number = int(group_match.group(1))
                    name = group_to_name.get(number)
                    return group_match.group(0) if name is None else f"- {name} (group{number}):"

                body = re.sub(
                    r"^-+\s*Group\s*(\d+)\s*:",
                    replace_group,
                    block_match.group(2),
                    flags=re.IGNORECASE | re.MULTILINE,
                )
                return block_match.group(1) + body

            output = block_pattern.sub(replace_block, output)
        return output


    shap_trend_text_inner = apply_group_mapping(category_mapping_text, trend_text)


    ###Load the table and add to database
    df0 = pd.read_csv(data_csv_path, index_col=False)
    df0.columns = [x.lower() for x in df0.columns]
    cat_cols = ['surfactant_in_water', 'oil_type', 'surfactant_in_oil']
    num_cols = ['ratio_of_surfactant_in_water', 'ratio_of_surfactant_in_oil', 'ion_concentration_in_water']
    target = 'reward'
    id_feats = ['candidateID']


    df0 = process_data(df=df0, cat_cols=cat_cols, num_cols=num_cols, id_feats=id_feats, target=target)

    ##Creating Data Summary
    data_summary = (generate_text_summary(df=df0,
                                cat_feats=[col.lower() for col in cat_cols],
                                num_feats=[col.lower() for col in num_cols]))

    docs_list = [data_summary, shap_text_summary, shap_trend_text_inner, category_mapping_text]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500, chunk_overlap=50
    )
    docs_list = [Document(page_content=text) for text in docs_list]
    documents = text_splitter.split_documents(docs_list)

    # Add to vectorDB
    vectorstore = Chroma.from_documents(
        documents=documents,
        collection_name="reward-rag-chroma-1",
        embedding=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY),

    )


    template = """
You are an assistant to chemist who study Magnatic-nano particles and water-oil interface . Answer the question based on the context below to help the agent.

Context: {context}

Question: {question}
"""

    prompt = ChatPromptTemplate.from_template(template)
    # llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-4-turbo")
    llm = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model="gpt-5",
        temperature=1
    )
    chain = (
        {"context": vectorstore.as_retriever(), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


    trend_analysis_prompt = f"""
You are an assistant to a materials chemist studying magnetic nanoparticles (MNPs) traversing oil-water interfaces.

You have access to a SHAP-based analysis that summarizes how six specific experimental variables influence the predicted reward.
This reward quantifies the MNP's ability to cross the oil-water interface while maintaining aggregation and uniformity.

SHAP values reflect the impact of each variable's value on the predicted reward:
- Positive SHAP values → increase reward (improved performance)
- Negative SHAP values → reduce reward (worse performance)

You must only consider the following six variables when analyzing SHAP trends and designing experiments:

1. surfactant_in_water (categorical)
2. oil_type (categorical)
3. surfactant_in_oil (categorical)
4. ratio_of_surfactant_in_water (numeric)
5. ratio_of_surfactant_in_oil (numeric)
6. ion_concentration_in_water (numeric)

Below is a SHAP-based multimodal analysis summarizing observed trends across different variable intervals or categories, along with categorical variable mapping information.
Use all provided context to interpret group numbers and to reason about SHAP trends.

The multimodal SHAP data include image-derived qualitative attributes.
Use these descriptors as physical indicators of interfacial transport efficiency and particle morphology.

- **interface_cross=yes** → MNPs successfully penetrate the water-oil interface and enter the oil phase,
  rather than being trapped at the boundary. Indicates effective crossing behavior.

- **uniformity=yes** → Particles in the oil phase remain well aggregated and form spherical clusters with good uniformity.
  If uniformity=no, particles show tailing or dispersion, implying unstable aggregation and increased residue.

- **residue=very little** → Quantifies how much particle residue remains in the oil phase after crossing.
  "Extensive" residue means poor transfer completeness, while "very little" means clean transition.

- **satellite=satellite_cluster_ratio** → Represents the fraction of small residual particle clusters relative to the total.
  A larger satellite ratio indicates stronger residual presence and less clean crossing.

You must integrate these visual cues when interpreting SHAP trends
{shap_trend_text_inner}

Your task is to identify 3 clear non-linear associations between experimental variables and MNP interfacial transfer performance.

You must output EXACTLY 3 associations in the following strict format. Do not include any preamble, introduction, summary, or extra text outside these 3 blocks:

ASSOCIATION 1: <one concise paragraph describing the non-linear relationship and its morphological interpretation>

ASSOCIATION 2: <one concise paragraph describing the non-linear relationship and its morphological interpretation>

ASSOCIATION 3: <one concise paragraph describing the non-linear relationship and its morphological interpretation>

Rules:
- Each ASSOCIATION block must be separated by exactly one blank line.
- Focus on scientific reasoning and clarity.
- Do not mention SHAP values.
- Do not invent new variables or combine existing ones. Stick strictly to the six given variables.
"""


    experiment_recommendation = chain.invoke(trend_analysis_prompt)


    def extract_associations(response: str) -> list[str]:
        """Extract association blocks from LLM response, falling back gracefully."""
        response = response.strip()
        # Parse strict format: ASSOCIATION 1:, ASSOCIATION 2:, ASSOCIATION 3:
        parts = re.split(
            r"(?im)^\s*ASSOCIATION\s+\d+\s*:\s*",
            response,
        )
        associations = [part.strip() for part in parts[1:] if part.strip()]
        if associations:
            return associations[:3]

        # Fallback: split by numbered patterns (1), 2., etc.)
        parts = re.split(
            r"(?m)^\s*(?:#{1,6}\s*)?(?:\*\*)?\d+[.):-]\s*(?:\*\*)?",
            response,
        )
        associations = [part.strip() for part in parts[1:] if part.strip()]
        if associations:
            return associations[:3]

        # Last resort: split by blank lines
        associations = [part.strip() for part in re.split(r"\n\s*\n", response) if part.strip()]
        if associations:
            return associations[:3]

        # Ultimate fallback: whole response as one association
        return [response]


    associations = extract_associations(experiment_recommendation)
    Path(Path(__file__).resolve().parent / "associations.json").write_text(
        json.dumps({"associations": associations}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("[LLM EXPERIMENT DESIGN BASED ON SHAP TRENDS]:\n" + "\n\n".join(associations))
    return associations


def run(state: dict) -> dict:
    """LangGraph node: Extraction Agent.
    Input: state with optional 'data_csv', 'caption_csv' keys
    Output: state updated with 'associations' list and 'associations_path'
    """
    project_dir = Path(__file__).resolve().parent
    data_csv = state.get("data_csv", str(project_dir / "augmented_data_with_pic.csv"))
    caption_csv = state.get("caption_csv", str(project_dir / "caption.csv"))
    associations = _run_extraction(data_csv, caption_csv)
    state["associations"] = associations
    state["associations_path"] = str(project_dir / "associations.json")
    state["data_csv"] = data_csv
    state["caption_csv"] = caption_csv
    return state


if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parent
    _run_extraction(
        str(project_dir / "augmented_data_with_pic.csv"),
        str(project_dir / "caption.csv")
    )
