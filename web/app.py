"""
Dubai Real Estate Price Predictor - Streamlit Web Application

This app uses the custom BaggingRegressor (implemented from scratch with PySpark)
to predict real estate prices based on property characteristics.

Run with: streamlit run app.py
"""

import streamlit as st
import json
import sys
from pathlib import Path
from datetime import date

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Configuration
APP_DIR = Path(__file__).parent
MODEL_DIR = APP_DIR / 'spark_model'
OPTIONS_PATH = APP_DIR / 'feature_options.json'
CONFIG_PATH = APP_DIR / 'model_config.json'


@st.cache_resource
def get_spark_session():
    """Create and return a Spark session."""
    from pyspark.sql import SparkSession
    spark = SparkSession.builder \
        .appName("bigboyz-web-predictor") \
        .config("spark.driver.memory", "2g") \
        .config("spark.ui.showConsoleProgress", "false") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


@st.cache_resource
def load_model():
    """Load the custom BaggingRegressor model."""
    from bagging_ensemble import load_bagging_model
    model_path = str(MODEL_DIR / 'bagging_model')
    return load_bagging_model(model_path)


@st.cache_resource
def load_pipeline():
    """Load the feature pipeline."""
    from pyspark.ml import PipelineModel
    pipeline_path = str(MODEL_DIR / 'feature_pipeline')
    return PipelineModel.load(pipeline_path)


@st.cache_data
def load_options():
    """Load feature options for dropdowns."""
    with open(OPTIONS_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


@st.cache_data
def load_config():
    """Load model configuration."""
    with open(CONFIG_PATH, 'r') as f:
        return json.load(f)


def format_currency(value):
    """Format value as AED currency."""
    return f"AED {value:,.0f}"


def predict_price(spark, pipeline_model, bagging_model, inputs):
    """
    Make a price prediction using the custom BaggingRegressor.

    Args:
        spark: Spark session
        pipeline_model: Fitted feature pipeline
        bagging_model: Trained BaggingRegressor
        inputs: Dictionary of user inputs

    Returns:
        Predicted price in AED
    """
    from pyspark.sql import Row
    from pyspark.sql.functions import lit

    # Extract temporal features from date
    trans_date = inputs['transaction_date']
    trans_year = trans_date.year
    trans_month = trans_date.month
    trans_quarter = (trans_month - 1) // 3 + 1
    trans_dayofweek = trans_date.isoweekday()  # 1=Monday, 7=Sunday

    # Create a single-row DataFrame with the input features
    row = Row(
        procedure_area=float(inputs['property_size']),
        has_parking=int(inputs['has_parking']),
        trans_year=trans_year,
        trans_month=trans_month,
        trans_quarter=trans_quarter,
        trans_dayofweek=trans_dayofweek,
        property_type_en=inputs['property_type'],
        property_usage_en=inputs['property_usage'],
        area_name_en=inputs['area'],
        nearest_metro_en=inputs['nearest_metro'],
        nearest_mall_en=inputs['nearest_mall']
    )

    input_df = spark.createDataFrame([row])

    # Transform through the feature pipeline
    features_df = pipeline_model.transform(input_df)

    # Make prediction using custom BaggingRegressor
    predictions_df = bagging_model.predict(features_df)

    # Get the prediction value
    prediction = predictions_df.select('prediction').collect()[0][0]

    return max(0, prediction)


def main():
    # Page configuration
    st.set_page_config(
        page_title="Dubai Real Estate Price Predictor",
        page_icon="🏠",
        layout="centered"
    )

    # Check if model files exist
    if not MODEL_DIR.exists() or not OPTIONS_PATH.exists():
        st.error("Model files not found! Please run `python train_model.py` first to train the model.")
        st.info("This will train the custom BaggingRegressor (from scratch) using PySpark.")
        st.stop()

    # Header
    st.title("🏠 Dubai Real Estate Price Predictor")
    st.markdown("""
    Predict real estate transaction prices in Dubai based on property characteristics,
    location, and temporal factors.

    **Model:** Custom BaggingRegressor implemented FROM SCRATCH with PySpark
    *Trained on 1.2M+ Dubai Land Department transactions*
    """)

    st.divider()

    # Load resources with spinner
    with st.spinner("Loading Spark and model (first load may take a moment)..."):
        try:
            spark = get_spark_session()
            pipeline_model = load_pipeline()
            bagging_model = load_model()
            options = load_options()
            config = load_config()
        except Exception as e:
            st.error(f"Failed to load model: {str(e)}")
            st.info("Please ensure you have run `python train_model.py` first.")
            st.stop()

    # Input form
    st.subheader("Property Details")

    col1, col2 = st.columns(2)

    with col1:
        # Property size
        property_size = st.number_input(
            "Property Size (sqm)",
            min_value=1.0,
            max_value=100000.0,
            value=150.0,
            step=10.0,
            help="The total area of the property in square meters"
        )

        # Property type
        property_type = st.selectbox(
            "Property Type",
            options=options['property_type_en'],
            index=options['property_type_en'].index('Unit') if 'Unit' in options['property_type_en'] else 0,
            help="Type of property (Villa, Land, Building, Unit)"
        )

        # Property usage
        property_usage = st.selectbox(
            "Property Usage",
            options=options['property_usage_en'],
            index=options['property_usage_en'].index('Residential') if 'Residential' in options['property_usage_en'] else 0,
            help="Intended use of the property"
        )

        # Has parking
        has_parking = st.checkbox(
            "Has Parking",
            value=True,
            help="Whether the property has parking facilities"
        )

    with col2:
        # Area
        default_area_idx = 0
        if 'Marsa Dubai' in options['area_name_en']:
            default_area_idx = options['area_name_en'].index('Marsa Dubai')
        elif 'Business Bay' in options['area_name_en']:
            default_area_idx = options['area_name_en'].index('Business Bay')

        area = st.selectbox(
            "Area",
            options=options['area_name_en'],
            index=default_area_idx,
            help="Dubai area/neighborhood"
        )

        # Nearest metro
        nearest_metro = st.selectbox(
            "Nearest Metro Station",
            options=options['nearest_metro_en'],
            index=0,
            help="The nearest metro station to the property"
        )

        # Nearest mall
        nearest_mall = st.selectbox(
            "Nearest Mall",
            options=options['nearest_mall_en'],
            index=0,
            help="The nearest shopping mall to the property"
        )

        # Transaction date
        transaction_date = st.date_input(
            "Transaction Date",
            value=date.today(),
            min_value=date(2004, 1, 1),
            max_value=date(2030, 12, 31),
            help="Expected date of transaction"
        )

    st.divider()

    # Predict button
    if st.button("🔮 Predict Price", type="primary", use_container_width=True):
        # Gather inputs
        inputs = {
            'property_size': property_size,
            'has_parking': 1 if has_parking else 0,
            'property_type': property_type,
            'property_usage': property_usage,
            'area': area,
            'nearest_metro': nearest_metro,
            'nearest_mall': nearest_mall,
            'transaction_date': transaction_date
        }

        # Make prediction
        with st.spinner("Running prediction through custom BaggingRegressor..."):
            try:
                predicted_price = predict_price(spark, pipeline_model, bagging_model, inputs)

                # Display result
                st.success("Prediction Complete!")

                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.metric(
                        label="Estimated Price",
                        value=format_currency(predicted_price),
                        delta=None
                    )

                # Price per sqm
                price_per_sqm = predicted_price / property_size if property_size > 0 else 0
                st.info(f"📐 Price per sqm: {format_currency(price_per_sqm)}")

                # Model info
                with st.expander("Model Information"):
                    st.markdown(f"""
                    - **Model Type:** {config['model_type']}
                    - **Implementation:** Custom from scratch (NOT using RandomForestRegressor)
                    - **Number of Trees:** {config['n_estimators']}
                    - **Max Tree Depth:** {config['max_depth']}
                    - **Test Set R²:** {config['test_metrics']['r2']:.4f}
                    - **Test Set RMSE:** {format_currency(config['test_metrics']['rmse'])}
                    - **Test Set MAE:** {format_currency(config['test_metrics']['mae'])}

                    The model uses:
                    - Bootstrap sampling (with replacement) for each tree
                    - DecisionTreeRegressor from MLlib as base learners
                    - Averaging of predictions across all trees

                    This implementation follows the CSCI316 assignment requirement to build
                    the bagging ensemble FROM SCRATCH without using RandomForestRegressor.
                    """)

            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: gray; font-size: 0.8em;'>
    CSCI316: Big Data Mining - Dubai Real Estate Price Prediction<br>
    University of Wollongong in Dubai<br>
    <br>
    Custom BaggingRegressor implemented from scratch using PySpark
    </div>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
