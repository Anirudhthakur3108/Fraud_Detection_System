import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .fraud-alert {
        background-color: #ff4b4b;
        color: white;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.2rem 0;
    }
    .safe-transaction {
        background-color: #00cc88;
        color: white;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.2rem 0;
    }
</style>
""", unsafe_allow_html=True)

class StreamlitFraudDetector:
    def __init__(self):
        self.isolation_forest = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.location_history = set()
        self.device_history = set()
        
    def preprocess_data(self, df):
        """Phase 1: Preprocessing"""
        try:
            # Convert dates
            df['TransactionDate'] = pd.to_datetime(df['TransactionDate'], format='ISO8601')
            df['PreviousTransactionDate'] = pd.to_datetime(df['PreviousTransactionDate'], format='ISO8601')
            
            # Extract datetime features
            df['Hour'] = df['TransactionDate'].dt.hour
            df['DayOfWeek'] = df['TransactionDate'].dt.dayofweek
            df['Month'] = df['TransactionDate'].dt.month
            df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
            df['IsNightTime'] = ((df['Hour'] >= 22) | (df['Hour'] <= 6)).astype(int)
            
            # Time since last transaction
            df['HoursSinceLastTransaction'] = (df['TransactionDate'] - df['PreviousTransactionDate']).dt.total_seconds() / 3600
            
            # Encode categorical variables
            categorical_columns = ['TransactionType', 'Location', 'Channel', 'CustomerOccupation']
            
            for col in categorical_columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col])
                else:
                    df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col])
            
            # Derived features
            df['AmountToBalanceRatio'] = df['TransactionAmount'] / (df['AccountBalance'] + 1)
            df['IsLargeAmount'] = (df['TransactionAmount'] > df['TransactionAmount'].quantile(0.95)).astype(int)
            df['IsLowBalance'] = (df['AccountBalance'] < df['AccountBalance'].quantile(0.25)).astype(int)
            
            # Store history
            self.location_history.update(df['Location'].unique())
            self.device_history.update(df['DeviceID'].unique())
            
            return df
        except Exception as e:
            st.error(f"Error in preprocessing: {e}")
            return None
    
    def ml_anomaly_detection(self, df, contamination=0.1):
        """Phase 2: ML-Based Anomaly Detection"""
        ml_features = [
            'TransactionAmount', 'CustomerAge', 'TransactionDuration', 'LoginAttempts',
            'AccountBalance', 'Hour', 'DayOfWeek', 'Month', 'IsWeekend', 'IsNightTime',
            'HoursSinceLastTransaction', 'AmountToBalanceRatio', 'IsLargeAmount', 'IsLowBalance',
            'TransactionType_encoded', 'Location_encoded', 'Channel_encoded', 'CustomerOccupation_encoded'
        ]
        
        # Handle missing values
        df[ml_features] = df[ml_features].fillna(df[ml_features].median())
        
        # Scale features
        X_scaled = self.scaler.fit_transform(df[ml_features])
        
        # Train Isolation Forest
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        
        self.isolation_forest.fit(X_scaled)
        
        # Get scores
        anomaly_scores = self.isolation_forest.decision_function(X_scaled)
        ml_risk_scores = (anomaly_scores.max() - anomaly_scores) / (anomaly_scores.max() - anomaly_scores.min())
        
        return ml_risk_scores
    
    def rule_based_scoring(self, df):
        """Phase 3: Rule-Based Risk Scoring"""
        rule_scores = []
        
        for _, row in df.iterrows():
            score = 0
            
            # Rule 1: High amount & low account balance
            if row['TransactionAmount'] > 500 and row['AccountBalance'] < 1000:
                score += 0.3
            
            # Rule 2: Very high transaction amount
            if row['TransactionAmount'] > df['TransactionAmount'].quantile(0.95):
                score += 0.2
            
            # Rule 3: Late-night transactions
            if row['IsNightTime'] == 1:
                score += 0.15
            
            # Rule 4: High login attempts
            if row['LoginAttempts'] > 1:
                score += 0.1 * (row['LoginAttempts'] - 1)
            
            # Rule 5: Long transaction duration
            if row['TransactionDuration'] > df['TransactionDuration'].quantile(0.9):
                score += 0.1
            
            # Rule 6: High amount to balance ratio
            if row['AmountToBalanceRatio'] > 0.5:
                score += 0.2
            
            # Rule 7: Weekend transactions
            if row['IsWeekend'] == 1:
                score += 0.05
            
            # Rule 8: Quick successive transactions
            if row['HoursSinceLastTransaction'] < 0.5:
                score += 0.1
            
            # Rule 9: Large amount for young customers
            if row['CustomerAge'] < 25 and row['TransactionAmount'] > 300:
                score += 0.1
            
            # Rule 10: High-risk online transactions
            if row['Channel'] == 'Online' and row['TransactionAmount'] > 400:
                score += 0.1
            
            rule_scores.append(min(score, 1.0))
        
        return np.array(rule_scores)
    
    def combine_scores(self, ml_scores, rule_scores, ml_weight=0.6, rule_weight=0.4, threshold=0.6):
        """Phase 4: Combine ML and Rule Scores"""
        # Normalize scores
        ml_scores_norm = (ml_scores - ml_scores.min()) / (ml_scores.max() - ml_scores.min())
        rule_scores_norm = (rule_scores - rule_scores.min()) / (rule_scores.max() - rule_scores.min())
        
        # Combine scores
        combined_scores = ml_weight * ml_scores_norm + rule_weight * rule_scores_norm
        
        # Make predictions
        fraud_predictions = (combined_scores > threshold).astype(int)
        
        return combined_scores, fraud_predictions, ml_scores_norm, rule_scores_norm

def main():
    st.markdown('<h1 class="main-header">🛡️ Advanced Fraud Detection System</h1>', unsafe_allow_html=True)
    
    # Initialize the detector
    if 'detector' not in st.session_state:
        st.session_state.detector = StreamlitFraudDetector()
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV file",
        type=['csv'],
        help="Upload your transaction data CSV file"
    )
    
    # Parameters
    st.sidebar.subheader("Model Parameters")
    contamination = st.sidebar.slider("Contamination Rate", 0.05, 0.2, 0.1, 0.01)
    ml_weight = st.sidebar.slider("ML Weight", 0.1, 0.9, 0.6, 0.1)
    rule_weight = 1 - ml_weight
    threshold = st.sidebar.slider("Fraud Threshold", 0.3, 0.8, 0.6, 0.05)
    
    st.sidebar.metric("Rule Weight", f"{rule_weight:.1f}")
    
    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ Data loaded successfully: {len(df)} transactions")
            
            # Display data preview
            with st.expander("📊 Data Preview"):
                st.dataframe(df.head())
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Transactions", len(df))
                with col2:
                    st.metric("Total Amount", f"${df['TransactionAmount'].sum():,.2f}")
                with col3:
                    st.metric("Avg Amount", f"${df['TransactionAmount'].mean():.2f}")
            
            # Process data
            if st.button("🚀 Run Fraud Detection", type="primary"):
                with st.spinner("Processing data..."):
                    
                    # Phase 1: Preprocessing
                    progress_bar = st.progress(0)
                    st.write("🔄 Phase 1: Preprocessing...")
                    df_processed = st.session_state.detector.preprocess_data(df)
                    progress_bar.progress(25)
                    
                    if df_processed is not None:
                        # Phase 2: ML Detection
                        st.write("🤖 Phase 2: ML-Based Anomaly Detection...")
                        ml_scores = st.session_state.detector.ml_anomaly_detection(df_processed, contamination)
                        progress_bar.progress(50)
                        
                        # Phase 3: Rule-based Scoring
                        st.write("📋 Phase 3: Rule-Based Risk Scoring...")
                        rule_scores = st.session_state.detector.rule_based_scoring(df_processed)
                        progress_bar.progress(75)
                        
                        # Phase 4: Combine Scores
                        st.write("🔄 Phase 4: Combining Scores...")
                        combined_scores, fraud_predictions, ml_scores_norm, rule_scores_norm = st.session_state.detector.combine_scores(
                            ml_scores, rule_scores, ml_weight, rule_weight, threshold
                        )
                        progress_bar.progress(100)
                        
                        # Results
                        st.success("✅ Fraud Detection Completed!")
                        
                        # Add results to dataframe
                        df_results = df_processed.copy()
                        df_results['ML_Score'] = ml_scores_norm
                        df_results['Rule_Score'] = rule_scores_norm
                        df_results['Combined_Score'] = combined_scores
                        df_results['Fraud_Prediction'] = fraud_predictions
                        
                        # Display metrics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric(
                                "Total Transactions",
                                len(df_results),
                                help="Total number of transactions processed"
                            )
                        
                        with col2:
                            fraud_count = sum(fraud_predictions)
                            st.metric(
                                "Flagged as Fraud",
                                fraud_count,
                                help="Number of transactions flagged as potentially fraudulent"
                            )
                        
                        with col3:
                            fraud_rate = (fraud_count / len(df_results)) * 100
                            st.metric(
                                "Fraud Rate",
                                f"{fraud_rate:.1f}%",
                                help="Percentage of transactions flagged as fraud"
                            )
                        
                        with col4:
                            avg_score = combined_scores.mean()
                            st.metric(
                                "Avg Risk Score",
                                f"{avg_score:.3f}",
                                help="Average combined risk score across all transactions"
                            )
                        
                        # Visualizations
                        st.subheader("📈 Analysis & Visualizations")
                        
                        # Score distribution
                        fig_scores = make_subplots(
                            rows=2, cols=2,
                            subplot_titles=('Combined Scores', 'ML Scores', 'Rule Scores', 'Score Comparison'),
                            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                                   [{"secondary_y": False}, {"secondary_y": False}]]
                        )
                        
                        # Combined scores histogram
                        fig_scores.add_trace(
                            go.Histogram(x=combined_scores, name="Combined", opacity=0.7, nbinsx=30),
                            row=1, col=1
                        )
                        
                        # ML scores histogram
                        fig_scores.add_trace(
                            go.Histogram(x=ml_scores_norm, name="ML", opacity=0.7, nbinsx=30),
                            row=1, col=2
                        )
                        
                        # Rule scores histogram
                        fig_scores.add_trace(
                            go.Histogram(x=rule_scores_norm, name="Rule", opacity=0.7, nbinsx=30),
                            row=2, col=1
                        )
                        
                        # Score comparison scatter
                        fig_scores.add_trace(
                            go.Scatter(
                                x=ml_scores_norm,
                                y=rule_scores_norm,
                                mode='markers',
                                name="ML vs Rule",
                                marker=dict(
                                    color=combined_scores,
                                    colorscale='Viridis',
                                    showscale=True,
                                    colorbar=dict(title="Combined Score")
                                )
                            ),
                            row=2, col=2
                        )
                        
                        fig_scores.update_layout(height=600, title_text="Risk Score Distributions")
                        st.plotly_chart(fig_scores, use_container_width=True)
                        
                        # Fraud vs Safe transactions
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Transaction amounts by fraud status
                            fig_amounts = px.box(
                                df_results,
                                x='Fraud_Prediction',
                                y='TransactionAmount',
                                title='Transaction Amounts: Fraud vs Safe',
                                labels={'Fraud_Prediction': 'Fraud (1) vs Safe (0)'}
                            )
                            st.plotly_chart(fig_amounts, use_container_width=True)
                        
                        with col2:
                            # Fraud by hour
                            fraud_by_hour = df_results.groupby('Hour')['Fraud_Prediction'].mean().reset_index()
                            fig_hour = px.bar(
                                fraud_by_hour,
                                x='Hour',
                                y='Fraud_Prediction',
                                title='Fraud Rate by Hour of Day',
                                labels={'Fraud_Prediction': 'Fraud Rate'}
                            )
                            st.plotly_chart(fig_hour, use_container_width=True)
                        
                        # Flagged transactions
                        st.subheader("🚨 Flagged Transactions")
                        
                        fraud_transactions = df_results[df_results['Fraud_Prediction'] == 1].sort_values('Combined_Score', ascending=False)
                        
                        if len(fraud_transactions) > 0:
                            st.write(f"Found {len(fraud_transactions)} potentially fraudulent transactions:")
                            
                            # Display top fraud transactions
                            display_cols = [
                                'TransactionID', 'TransactionAmount', 'AccountBalance', 
                                'Location', 'TransactionDate', 'Combined_Score', 
                                'ML_Score', 'Rule_Score'
                            ]
                            
                            fraud_display = fraud_transactions[display_cols].round(3)
                            
                            # Color code the dataframe
                            def highlight_fraud(val):
                                if isinstance(val, (int, float)) and val > threshold:
                                    return 'background-color: #ffcccc'
                                return ''
                            
                            styled_df = fraud_display.style.applymap(highlight_fraud, subset=['Combined_Score'])
                            st.dataframe(styled_df, use_container_width=True)
                            
                            # Detailed analysis for top fraud cases
                            st.subheader("🔍 Top Risk Transactions Analysis")
                            
                            top_n = st.selectbox("Show top N risky transactions:", [5, 10, 15, 20], index=0)
                            
                            for i, (idx, row) in enumerate(fraud_transactions.head(top_n).iterrows()):
                                with st.expander(f"🚨 Transaction {row['TransactionID']} - Risk Score: {row['Combined_Score']:.3f}"):
                                    col1, col2, col3 = st.columns(3)
                                    
                                    with col1:
                                        st.write("**Transaction Details:**")
                                        st.write(f"Amount: ${row['TransactionAmount']:,.2f}")
                                        st.write(f"Balance: ${row['AccountBalance']:,.2f}")
                                        st.write(f"Location: {row['Location']}")
                                        st.write(f"Channel: {row['Channel']}")
                                        st.write(f"Date: {row['TransactionDate']}")
                                    
                                    with col2:
                                        st.write("**Customer Info:**")
                                        st.write(f"Age: {row['CustomerAge']}")
                                        st.write(f"Occupation: {row['CustomerOccupation']}")
                                        st.write(f"Login Attempts: {row['LoginAttempts']}")
                                        st.write(f"Transaction Duration: {row['TransactionDuration']}s")
                                    
                                    with col3:
                                        st.write("**Risk Analysis:**")
                                        st.write(f"Combined Score: {row['Combined_Score']:.3f}")
                                        st.write(f"ML Score: {row['ML_Score']:.3f}")
                                        st.write(f"Rule Score: {row['Rule_Score']:.3f}")
                                        st.write(f"Amount/Balance Ratio: {row['AmountToBalanceRatio']:.3f}")
                                        
                                        if row['IsNightTime']:
                                            st.write("⚠️ Night-time transaction")
                                        if row['IsWeekend']:
                                            st.write("⚠️ Weekend transaction")
                        else:
                            st.info("No transactions flagged as fraudulent with current threshold.")
                        
                        # Feature importance analysis
                        st.subheader("📊 Feature Analysis")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Channel distribution
                            channel_fraud = df_results.groupby(['Channel', 'Fraud_Prediction']).size().unstack(fill_value=0)
                            if 1 in channel_fraud.columns:
                                channel_fraud['Fraud_Rate'] = channel_fraud[1] / (channel_fraud[0] + channel_fraud[1])
                                fig_channel = px.bar(
                                    channel_fraud.reset_index(),
                                    x='Channel',
                                    y='Fraud_Rate',
                                    title='Fraud Rate by Channel'
                                )
                                st.plotly_chart(fig_channel, use_container_width=True)
                        
                        with col2:
                            # Location distribution
                            location_fraud = df_results.groupby('Location')['Fraud_Prediction'].agg(['count', 'sum', 'mean']).reset_index()
                            location_fraud.columns = ['Location', 'Total_Transactions', 'Fraud_Count', 'Fraud_Rate']
                            location_fraud = location_fraud[location_fraud['Total_Transactions'] >= 2]  # Filter locations with at least 2 transactions
                            
                            if len(location_fraud) > 0:
                                fig_location = px.scatter(
                                    location_fraud,
                                    x='Total_Transactions',
                                    y='Fraud_Rate',
                                    size='Fraud_Count',
                                    hover_data=['Location'],
                                    title='Fraud Rate by Location'
                                )
                                st.plotly_chart(fig_location, use_container_width=True)
                        
                        # Download results
                        st.subheader("💾 Download Results")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Full results CSV
                            csv_full = df_results.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Full Results (CSV)",
                                data=csv_full,
                                file_name=f"fraud_detection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                        
                        with col2:
                            # Fraud-only CSV
                            if len(fraud_transactions) > 0:
                                csv_fraud = fraud_transactions.to_csv(index=False)
                                st.download_button(
                                    label="🚨 Download Fraud Cases (CSV)",
                                    data=csv_fraud,
                                    file_name=f"fraud_cases_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv"
                                )
                        
                        # Summary report
                        st.subheader("📋 Summary Report")
                        
                        summary_report = f"""
                        ## Fraud Detection Summary Report
                        **Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                        
                        ### Dataset Overview
                        - **Total Transactions:** {len(df_results):,}
                        - **Date Range:** {df_results['TransactionDate'].min()} to {df_results['TransactionDate'].max()}
                        - **Total Transaction Volume:** ${df_results['TransactionAmount'].sum():,.2f}
                        
                        ### Model Configuration
                        - **ML Weight:** {ml_weight:.1f}
                        - **Rule Weight:** {rule_weight:.1f}
                        - **Fraud Threshold:** {threshold:.2f}
                        - **Contamination Rate:** {contamination:.2f}
                        
                        ### Detection Results
                        - **Transactions Flagged:** {fraud_count:,} ({fraud_rate:.2f}%)
                        - **Average Risk Score:** {avg_score:.3f}
                        - **High-Risk Transactions (>0.8):** {sum(combined_scores > 0.8):,}
                        - **Medium-Risk Transactions (0.5-0.8):** {sum((combined_scores >= 0.5) & (combined_scores <= 0.8)):,}
                        
                        ### Risk Distribution
                        - **ML Score Range:** {ml_scores_norm.min():.3f} - {ml_scores_norm.max():.3f}
                        - **Rule Score Range:** {rule_scores_norm.min():.3f} - {rule_scores_norm.max():.3f}
                        - **Combined Score Range:** {combined_scores.min():.3f} - {combined_scores.max():.3f}
                        
                        ### Recommendations
                        1. **High Priority:** Review all transactions with Combined Score > 0.8
                        2. **Medium Priority:** Monitor transactions with Combined Score 0.6-0.8
                        3. **Pattern Analysis:** Focus on {fraud_by_hour.loc[fraud_by_hour['Fraud_Prediction'].idxmax(), 'Hour']}:00 hour (highest fraud rate)
                        4. **Channel Focus:** Review security for channels with high fraud rates
                        """
                        
                        st.markdown(summary_report)
                        
                        # Save summary report
                        st.download_button(
                            label="📋 Download Summary Report (MD)",
                            data=summary_report,
                            file_name=f"fraud_summary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                            mime="text/markdown"
                        )
                
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.write("Please check your CSV file format and try again.")
    
    else:
        st.info("👆 Please upload a CSV file to begin fraud detection analysis.")
        
        # Show sample data format
        with st.expander("📋 Expected CSV Format"):
            st.write("Your CSV file should contain the following columns:")
            sample_columns = [
                'TransactionID', 'AccountID', 'TransactionAmount', 'TransactionDate',
                'TransactionType', 'Location', 'DeviceID', 'IP Address', 'MerchantID',
                'Channel', 'CustomerAge', 'CustomerOccupation', 'TransactionDuration',
                'LoginAttempts', 'AccountBalance', 'PreviousTransactionDate'
            ]
            
            for i, col in enumerate(sample_columns, 1):
                st.write(f"{i}. **{col}**")
            
            st.write("\n**Date Format:** DD-MM-YYYY HH:MM")
            st.write("**Example:** 11-04-2023 16:29")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; font-size: 0.8em;'>
            🛡️ Advanced Fraud Detection System | Built with Streamlit<br>
            For technical support or questions, please contact your system administrator.
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()