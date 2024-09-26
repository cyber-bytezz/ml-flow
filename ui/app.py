import streamlit as st
from model.inference import predict
from monitoring.metrics_visualizer import plot_metrics

def main():
    st.title("Text Classification Model")
    
    text_input = st.text_area("Enter text for prediction:")
    
    if st.button("Predict"):
        prediction = predict(text_input)
        st.write(f"Predicted Class: {prediction}")
    
    st.write("Model Performance Metrics")
    plot_metrics()

if __name__ == "__main__":
    main()
