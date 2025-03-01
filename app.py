import streamlit as st
import pandas as pd
import numpy as np
from exp_main import run

st.set_page_config(page_title="MMO-EvoBagging Dashboard", layout="wide")

def main():
    st.title("MMO-EvoBagging Dashboard")

    # Pipeline Controls in sidebar
    with st.sidebar:
        st.header("Pipeline Controls")
        dataset_name = st.selectbox("Dataset", ["pima", "breast_cancer", "ionosphere", "sonar", "heart"])
        test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.1)
        n_exp = st.slider("Number of Experiments", 1, 50, 1)
        n_bags = st.slider("Number of Bags", 2, 20, 8)
        n_iter = st.slider("Number of Iterations", 1, 50, 1)
        
        if st.button("Start Training"):
            with st.spinner("Training in progress..."):
                try:
                    # Create progress placeholder
                    progress_text = st.empty()
                    progress_text.text("Starting training...")
                    
                    # Run training
                    results = run(
                        dataset_name=dataset_name,
                        test_size=test_size,
                        n_exp=n_exp,
                        metric='accuracy',
                        n_bags=n_bags,
                        n_iter=n_iter,
                        n_select=2,
                        n_new_bags=2,
                        n_mutation=2,
                        mutation_rate=0.05,
                        size_coef=0.2,
                        clf_coef=0.2,
                        voting='majority',
                        n_test=8,
                        procs=4
                    )
                    
                    # Store results in session state
                    st.session_state.results = results
                    progress_text.text("Training completed!")
                    st.success("Training completed successfully!")
                    
                except Exception as e:
                    st.error(f"Error during training: {str(e)}")
                    st.write("Full error:", str(e))

    # Main content area
    st.header("Training Results")
    
    if 'results' in st.session_state:
        results = st.session_state.results
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Training Accuracy", f"{results.get('train_accuracy', 0):.2%}")
        with col2:
            st.metric("Testing Accuracy", f"{results.get('test_accuracy', 0):.2%}")
        with col3:
            st.metric("F1 Score", f"{results.get('f1_score', 0):.2%}")
        
        # Display detailed results
        st.subheader("Detailed Results")
        st.json(results)
        
    else:
        st.info("No results available. Start training to see results.")

    # About section
    st.sidebar.markdown("""
    ### About
    This application implements MMO-EvoBagging, a multi-objective optimization approach 
    for ensemble learning using bagging.

    ### Features
    - Multiple dataset support
    - Customizable parameters
    - Real-time training monitoring
    - Performance visualization
    """)

if __name__ == "__main__":
    main()