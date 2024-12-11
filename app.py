import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import subprocess
import os
import base64
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression


def desc_calc():
    """Performs the descriptor calculation using PaDEL-Descriptor"""
    bashCommand = "java -Xms2G -Xmx2G -Djava.awt.headless=true -jar ./PaDEL-Descriptor/PaDEL-Descriptor.jar -removesalt -standardizenitro -fingerprints -descriptortypes ./PaDEL-Descriptor/PubchemFingerprinter.xml -dir ./ -file descriptors_output.csv"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

    # Add these debug lines
    print("PaDEL Descriptor Calculation Output:", output.decode())
    if error:
        print("PaDEL Descriptor Calculation Error:", error.decode())

    os.remove("molecule.smi")

desc = pd.read_csv("descriptors_output.csv")


def filedownload(df):
    """Create a download link for the predictions"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Download Predictions</a>'
    return href

def prepare_features(desc_data, feature_names):
    """
    Prepare features for model prediction
    1. Remove non-numeric columns
    2. Scale features
    3. Select top features
    """
    try:
        
        numeric_df = desc_data.select_dtypes(include=[np.number])

        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(numeric_df)

        # Create a DataFrame with the scaled features
        scaled_df = pd.DataFrame(X_scaled, columns=numeric_df.columns)

        # Select only the features expected by the model
        selected_features_df = scaled_df[feature_names]

        return selected_features_df

    except Exception as e:
        st.error(f"Error in feature preparation: {str(e)}")
        return None


def build_model(desc_data, molecule_names):
    """Build and apply the prediction model"""
    try:
        with open("random_forest_model.pkl", "rb") as model_file:
            model_dict = pickle.load(model_file)
        load_model = model_dict["model"]  
        feature_names = model_dict["feature_names"]  

        # Prepare features with the correct feature names
        processed_features = prepare_features(desc_data, feature_names)

        if processed_features is None:
            st.error("Failed to process features")
            return None

        # Debugging:
        st.write("Processed feature names:", processed_features.columns.tolist())

        
        prediction = load_model.predict(processed_features)

        # Create output DataFrame
        prediction_output = pd.Series(prediction, name="pIC50")
        molecule_name = pd.Series(molecule_names, name="molecule_name")
        df = pd.concat([molecule_name, prediction_output], axis=1)

        # Display results
        st.header("**Prediction output**")
        st.write(df)
        st.markdown(filedownload(df), unsafe_allow_html=True)

        return df

    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        st.write("## Debug Information")
        st.write("Input data shape:", desc_data.shape)
        st.write("Molecule names:", molecule_names)
        return None


#################################################


def main():
    # Logo image
    image = Image.open("logo.png")
    st.image(image, use_container_width=True)

    # Page title
    st.markdown("""
    # Bioactivity Prediction App (Trypsin)
    This app allows you to predict the bioactivity towards inhibiting the Trypsin enzymes.
    """)

    # Sidebar
    st.sidebar.header("1. Upload your CSV data")
    uploaded_file = st.sidebar.file_uploader("Upload your input file", type=["txt"])

    if uploaded_file is not None:
        # Load and display input data
        load_data = pd.read_table(uploaded_file, sep=" ", header=None)

        if st.sidebar.button("Predict"):
            # Save to temporary file for PaDEL-Descriptor
            load_data.to_csv("molecule.smi", sep="\t", header=False, index=False)

            st.header("**Original input data**")
            st.write(load_data)

            # Calculate descriptors
            with st.spinner("Calculating descriptors..."):
                desc_calc()

            
            st.header("**Calculated molecular descriptors**")
            desc = pd.read_csv("descriptors_output.csv")
            st.write(desc)
            st.write(desc.shape)

            
            molecule_names = load_data[1].tolist()

            
            build_model(desc, molecule_names)
    else:
        st.info("Upload input data in the sidebar to start!")


if __name__ == "__main__":
    main()


with open("random_forest_model.pkl", "rb") as f:
    model = pickle.load(f)
