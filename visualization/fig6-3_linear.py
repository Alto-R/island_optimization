"""
Advanced Energy System Investment Analysis with Linear Regression Focus

This script implements a comprehensive analysis approach:
1. Correlation Analysis: Global view with correlation matrix heatmap and collinearity checks
2. Linear Modeling Baseline: Multiple linear regression as comparison benchmark  
3. Enhanced Linear Analysis: Detailed linear regression with feature importance
4. Advanced Correlation: Partial correlation analysis and stratified grouping

Author: Claude Code Assistant
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Core ML libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# SHAP analysis removed as per requirements

# Statistical libraries
from scipy.stats import pearsonr
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

import os
import glob


class AdvancedEnergyAnalysis:
    """
    Advanced analysis class implementing the four-stage progressive modeling approach
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.results = {}
        self.models = {}
        self.data = None
        self.feature_names = None
        self.target_names = None
        
        # Equipment repair times in hours
        self.repair_time_hours = {
            'WT': 336,   # Wind Turbine: 336 hours
            'PV': 96,    # PV: 96 hours  
            'WEC': 336,  # WEC: 336 hours
            'LNG': 96    # LNG/CHP: 96 hours
        }
        
        # Convert to 3-hour time steps for simulation
        self.repair_time_steps = {
            device: hours//3 for device, hours in self.repair_time_hours.items()
        }
        
        # Set Nature journal plotting style
        plt.style.use('default')
        plt.rcParams.update({
            'font.family': 'Arial',  # Nature prefers Arial/Helvetica
            'font.size': 10,
            'axes.labelsize': 10,
            'axes.titlesize': 11,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'axes.linewidth': 0.8,
            'grid.linewidth': 0.4,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'xtick.direction': 'out',
            'ytick.direction': 'out',
            'axes.edgecolor': '#333333',
            'text.color': '#333333',
            'axes.labelcolor': '#333333',
            'xtick.color': '#333333',
            'ytick.color': '#333333'
        })
    
    def failure_probability_wt(self, ve):
        """Wind Turbine failure probability based on wind speed"""
        if ve < 30:
            return 0
        elif ve >= 60:
            return 1.0
        else:
            return (ve - 30) / (60 - 30)
    
    def failure_probability_pv(self, ve):
        """PV failure probability based on wind speed"""
        if ve < 40:
            return 0
        elif ve >= 80:
            return 1.0
        else:
            return (ve - 40) / (80 - 40)
    
    def failure_probability_wec(self, ve):
        """WEC failure probability based on wind speed (simplified without wave height)"""
        if ve < 35:
            return 0
        elif ve >= 70:
            return 1.0
        else:
            return (ve - 35) / (70 - 35)
    
    def device_state_simulation(self, wind_speeds):
        """
        Simulate device states over time considering failure and repair dynamics
        """
        time_horizon = len(wind_speeds)
        device_generate = ['WT', 'PV', 'WEC', 'LNG']
        device_states_df = pd.DataFrame(index=range(time_horizon), columns=device_generate, dtype=int)
        
        device_states = {device: 1 for device in device_generate}  # 1 = working, 0 = failed
        time_in_states = {device: 0 for device in device_generate}
        
        np.random.seed(self.random_state)
        
        for t in range(time_horizon):
            V = wind_speeds.iloc[t] if hasattr(wind_speeds, 'iloc') else wind_speeds[t]
            
            for device in device_generate:
                if device == 'LNG':
                    # LNG equipment fails when wind speed > 20 m/s
                    if V > 20:
                        device_states[device] = 0
                        time_in_states[device] = 0  # Reset repair counter
                    else:
                        device_states[device] = 1
                else:
                    # Check for new failures in working equipment
                    if device_states[device] == 1:
                        if device == 'WT':
                            failure_prob = self.failure_probability_wt(V)
                        elif device == 'PV':
                            failure_prob = self.failure_probability_pv(V)
                        elif device == 'WEC':
                            failure_prob = self.failure_probability_wec(V)
                        else:
                            failure_prob = 0
                        
                        if np.random.random() < failure_prob:
                            device_states[device] = 0
                            time_in_states[device] = 0  # Reset repair counter
                
                    # Handle repair for failed equipment
                    if device_states[device] == 0:
                        # Can only repair when wind speed is suitable (≤ 20 m/s)
                        if V <= 20:
                            time_in_states[device] += 1
                        
                        # Check if repair is complete
                        if time_in_states[device] >= self.repair_time_steps[device]:
                            device_states[device] = 1  # Repair complete
                            time_in_states[device] = 0  # Reset counter
                
                # High wind speed forces all equipment to stop
                if V > 20:
                    device_states[device] = 0
                
                # Record current state
                device_states_df.at[t, device] = device_states[device]
        
        return device_states_df
    
    def extract_failure_features(self, scenario_df):
        """
        Extract comprehensive failure features from equipment failure simulation
        Returns all durations in hours (time_steps * 3)
        """
        features = {}
        devices = scenario_df.columns
        time_step_hours = 3  # Each time step represents 3 hours
        
        # Initialize aggregated features
        total_system_downtime_hours = 0
        max_system_downtime_hours = 0
        total_failure_events = 0
        
        for device in devices:
            downtime = 1 - scenario_df[device]  # 1 = downtime, 0 = working
            
            # Convert time steps to hours
            # Feature 1: Total downtime in hours
            total_downtime_steps = downtime.sum()
            total_downtime_hours = total_downtime_steps * time_step_hours
            features[f'total_downtime_hours_{device}'] = total_downtime_hours
            
            # Feature 2: Maximum consecutive downtime in hours
            consecutive_groups = (downtime != downtime.shift()).cumsum()
            consecutive_downtime = downtime.groupby(consecutive_groups).sum()
            max_consecutive_steps = consecutive_downtime.max() if len(consecutive_downtime) > 0 else 0
            max_consecutive_hours = max_consecutive_steps * time_step_hours
            features[f'max_consecutive_downtime_hours_{device}'] = max_consecutive_hours
            
            # Feature 3: Number of failure events
            failure_events = (scenario_df[device].diff() == -1).sum()
            features[f'failure_events_{device}'] = failure_events
            
            # Feature 4: Average downtime per event (hours)
            if failure_events > 0:
                avg_downtime_hours = total_downtime_hours / failure_events
                features[f'avg_downtime_per_event_hours_{device}'] = avg_downtime_hours
            else:
                features[f'avg_downtime_per_event_hours_{device}'] = 0
            
            # Feature 5: Downtime frequency (events per 1000 hours)
            total_simulation_hours = len(scenario_df) * time_step_hours
            if total_simulation_hours > 0:
                failure_frequency = (failure_events / total_simulation_hours) * 1000
                features[f'failure_frequency_per_1000h_{device}'] = failure_frequency
            else:
                features[f'failure_frequency_per_1000h_{device}'] = 0
            
            # Accumulate system-wide statistics
            total_system_downtime_hours += total_downtime_hours
            max_system_downtime_hours = max(max_system_downtime_hours, max_consecutive_hours)
            total_failure_events += failure_events
        
        # System-wide features
        # Feature 6: Multi-device simultaneous downtime in hours
        simultaneous_2_steps = (scenario_df.sum(axis=1) <= (len(devices) - 2)).sum()
        simultaneous_3_steps = (scenario_df.sum(axis=1) <= (len(devices) - 3)).sum()
        features['simultaneous_2_downtime_hours'] = simultaneous_2_steps * time_step_hours
        features['simultaneous_3_downtime_hours'] = simultaneous_3_steps * time_step_hours
        
        # Feature 7: System reliability metrics
        features['max_failure_duration_hours'] = max_system_downtime_hours  # Main feature for analysis
        features['total_system_downtime_hours'] = total_system_downtime_hours
        
        # Feature 8: System availability (removed)
        
        # Feature 9: Critical failure periods (removed)
        
        return features
    
    def calculate_real_data(self):
        """
        Calculate comprehensive island energy system data for both 2020 and 2050 scenarios
        """
        print("Loading and preparing island energy system data for 2020 and 2050 scenarios...")
        
        # Read island basic data
        island_data_path = '../result/island_data_origin.csv'
        if not os.path.exists(island_data_path):
            print(f"Error: {island_data_path} not found")
            return None
        
        islands_df = pd.read_csv(island_data_path)
        results_data = []
        
        # Define scenarios
        scenarios = {
            2020: {
                'output_dir': '../result/output_2020',
                'cost_summary_file': '../result/island_cost_summary_2020.csv'
            },
            2050: {
                'output_dir': '../result/output_2050',
                'cost_summary_file': '../result/island_cost_summary_2050.csv'
            }
        }
        
        # Common directories
        windspeed_dir = '../result/island_windspeed_data'
        demand_dir = '../demand_get/data/get1'
        
        # Get windspeed files for both years
        windspeed_files_2020 = glob.glob(os.path.join(windspeed_dir, '*_2020_windspeed.csv'))
        windspeed_files_2050 = glob.glob(os.path.join(windspeed_dir, '*_2050_windspeed.csv'))
        demand_files = glob.glob(os.path.join(demand_dir, 'demand_*.csv'))
        
        print(f"Found {len(windspeed_files_2020)} windspeed files for 2020")
        print(f"Found {len(windspeed_files_2050)} windspeed files for 2050")
        
        successful_matches = 0
        failed_matches = 0
        
        # Process data for each scenario (2020 and 2050)
        for year, config in scenarios.items():
            print(f"Processing {year} scenario...")
            
            output_dir = config['output_dir']
            cost_summary_file = config['cost_summary_file']
            
            # Get scenario-specific file lists
            output_files = glob.glob(os.path.join(output_dir, '*_results.csv'))
            capacity_files = glob.glob(os.path.join(output_dir, '*_capacity.csv'))
            
            # Load cost summary
            cost_summary_df = None
            if cost_summary_file and os.path.exists(cost_summary_file):
                cost_summary_df = pd.read_csv(cost_summary_file)
            
            for idx, island in islands_df.iterrows():
                lat, lon = island['Lat'], island['Long']
                
                # Find matching files - use appropriate year's windspeed data
                if year == 2020:
                    windspeed_file = next((f for f in windspeed_files_2020 if f"{lat}_{lon}_2020_windspeed.csv" in f), None)
                else:  # year == 2050
                    windspeed_file = next((f for f in windspeed_files_2050 if f"{lat}_{lon}_2050_windspeed.csv" in f), None)
                
                output_file = next((f for f in output_files if f"{lat}_{lon}_results.csv" in f), None)
                capacity_file = next((f for f in capacity_files if f"{lat}_{lon}_capacity.csv" in f), None)
                demand_file = next((f for f in demand_files if f"demand_{lat}_{lon}.csv" in f), None)
                
                # Find cost data
                cost_data = None
                if cost_summary_df is not None:
                    cost_matches = cost_summary_df[
                        (cost_summary_df['lat'] == lat) & (cost_summary_df['lon'] == lon)
                    ]
                    if len(cost_matches) > 0:
                        cost_data = cost_matches.iloc[0]
            
                if windspeed_file and output_file and capacity_file and cost_data is not None:
                    try:
                        # Process wind speed data for PDI calculation
                        wind_df = pd.read_csv(windspeed_file)
                        # Choose appropriate wind speed column based on year
                        if year == 2020:
                            wind_col = 'Wind_Speed_2020' if 'Wind_Speed_2020' in wind_df.columns else wind_df.columns[0]
                        else:  # year == 2050
                            wind_col = 'Wind_Speed_2050' if 'Wind_Speed_2050' in wind_df.columns else wind_df.columns[0]
                        
                        wind_speeds = wind_df[wind_col]
                        pdi = (wind_speeds ** 3).sum()  # Power Dissipation Index
                        
                        # Advanced equipment failure state simulation
                        device_states_df = self.device_state_simulation(wind_speeds)
                        failure_features = self.extract_failure_features(device_states_df)
                        
                        # Extract key reliability metrics
                        max_failure_duration = failure_features['max_failure_duration_hours']
                        
                        # Process energy output data
                        output_df = pd.read_csv(output_file)
                        e_wt = output_df['WT'].sum() if 'WT' in output_df.columns else 0
                        e_pv = output_df['PV'].sum() if 'PV' in output_df.columns else 0
                        e_wec = output_df['WEC'].sum() if 'WEC' in output_df.columns else 0
                        e_chp = output_df['CHP_electric_output'].sum() if 'CHP_electric_output' in output_df.columns else 0
                        
                        renewable_energy = e_wt + e_pv + e_wec
                        total_energy = renewable_energy + e_chp
                        renewable_penetration = renewable_energy / total_energy if total_energy > 0 else 0
                        
                        # Process demand data
                        cooling_demand = 0
                        heating_demand = 0
                        if demand_file and os.path.exists(demand_file):
                            try:
                                demand_df = pd.read_csv(demand_file)
                                cooling_demand = demand_df['cooling_demand'].sum() if 'cooling_demand' in demand_df.columns else 0
                                heating_demand = demand_df['heating_demand'].sum() if 'heating_demand' in demand_df.columns else 0
                            except:
                                pass
                        
                        # Process cost data
                        renewable_cost = cost_data.get('renewable_cost_per_capita', 0)
                        storage_investment = cost_data.get('storage_cost_per_capita', 0)
                        lng_cost = cost_data.get('lng_cost_per_capita', 0)
                        
                        results_data.append({
                            'ID': island['ID'],
                            'Long': lon,
                            'Lat': lat,
                            'Country': island['Country'],
                            'Island': island['Island'],
                            'Population': island['pop'],
                            'Year': year,  # Add year information to distinguish scenarios
                            # Independent variables (features)
                            'PDI': pdi,
                            'Max_Failure_Duration_Hours': max_failure_duration,  # Hours instead of time steps
                            'Renewable_Penetration': renewable_penetration,
                            'Cooling_Demand': cooling_demand,
                            'Heating_Demand': heating_demand,
                            # Dependent variables (targets)
                            'Renewable_Cost': renewable_cost,
                            'Total_Storage_Investment': storage_investment,
                            'LNG_Cost': lng_cost
                        })
                        successful_matches += 1
                        
                    except Exception as e:
                        failed_matches += 1
                        continue
                else:
                    failed_matches += 1
        
        # Create DataFrame
        analysis_df = pd.DataFrame(results_data)
        
        if len(analysis_df) == 0:
            print("❌ Error: No data available for analysis")
            return None
        
        # Basic filtering
        initial_count = len(analysis_df)
        analysis_df = analysis_df[
            (analysis_df['PDI'] > 0) & 
            (analysis_df['Max_Failure_Duration_Hours'] >= 0) &
            (analysis_df['Cooling_Demand'] >= 0) &
            (analysis_df['Heating_Demand'] >= 0) &
            (analysis_df['Renewable_Cost'] >= 0) &
            (analysis_df['Total_Storage_Investment'] >= 0) &
            (analysis_df['LNG_Cost'] >= 0)
        ]
        
        print(f"Data loaded successfully:")
        print(f"   - Total records: {initial_count}")
        print(f"   - After filtering: {len(analysis_df)}")
        print(f"   - Success rate: {successful_matches/(successful_matches+failed_matches)*100:.1f}%")
        print(f"   - Records per year: 2020({len(analysis_df[analysis_df['Year']==2020])}), 2050({len(analysis_df[analysis_df['Year']==2050])})")
        
        self.data = analysis_df
        # Features are physical quantities calculated from climate scenarios
        # Year is kept as metadata to distinguish scenarios but not used as a feature
        self.feature_names = [
            'PDI',                           # Power Dissipation Index
            'Max_Failure_Duration_Hours',    # Maximum failure duration in hours
        ]
        
        # Control variables for partial correlation analysis
        self.control_variables = [
            'Heating_Demand',               # Heating demand as control variable
            'Cooling_Demand',               # Cooling demand as control variable
        ]
        
        # Grouping variable
        self.grouping_variable = 'Renewable_Penetration'  # For renewable energy penetration grouping
        self.target_names = ['Renewable_Cost', 'Total_Storage_Investment', 'LNG_Cost']
        
        return analysis_df
    
    def apply_data_cleaning(self, strategy='moderate'):
        """
        Apply data cleaning strategy
        """
        if self.data is None:
            print("❌ Error: No data loaded")
            return None
        
        print(f"\n🔧 Applying {strategy.upper()} data cleaning strategy...")
        
        df_cleaned = self.data.copy()
        original_count = len(df_cleaned)
        
        if strategy == 'moderate':
            # Remove cases with all investment costs = 0
            mask_all_zero = (
                (df_cleaned['Renewable_Cost'] == 0) & 
                (df_cleaned['Total_Storage_Investment'] == 0) & 
                (df_cleaned['LNG_Cost'] == 0)
            )
            df_cleaned = df_cleaned[~mask_all_zero]
            
            # Remove inconsistent renewable data
            mask_renewable_inconsistent = (
                (df_cleaned['Renewable_Penetration'] == 0) & 
                (df_cleaned['Renewable_Cost'] == 0)
            )
            df_cleaned = df_cleaned[~mask_renewable_inconsistent]
        
        elif strategy == 'conservative':
            # Only remove clearly problematic cases
            mask_all_zero = (
                (df_cleaned['Renewable_Cost'] == 0) & 
                (df_cleaned['Total_Storage_Investment'] == 0) & 
                (df_cleaned['LNG_Cost'] == 0)
            )
            df_cleaned = df_cleaned[~mask_all_zero]
        
        elif strategy == 'aggressive':
            # Remove most zeros
            mask_any_zero_cost = (
                (df_cleaned['Renewable_Cost'] == 0) | 
                (df_cleaned['Total_Storage_Investment'] == 0) | 
                (df_cleaned['LNG_Cost'] == 0)
            )
            df_cleaned = df_cleaned[~mask_any_zero_cost]
            df_cleaned = df_cleaned[df_cleaned['Renewable_Penetration'] > 0]
        
        final_count = len(df_cleaned)
        removed_count = original_count - final_count
        
        print(f"   • Original records: {original_count}")
        print(f"   • Final records: {final_count}")
        print(f"   • Removed: {removed_count} ({removed_count/original_count*100:.1f}%)")
        
        self.data = df_cleaned
        return df_cleaned

    def stage1_correlation_analysis(self):
        """
        Stage 1: Correlation Analysis - Global view with heatmap and collinearity checks
        """
        print("\n" + "="*80)
        print("🎯 STAGE 1: CORRELATION ANALYSIS")
        print("="*80)
        
        if self.data is None:
            print("❌ Error: No data available")
            return
        
        # Prepare data
        all_vars = self.feature_names + self.target_names
        corr_data = self.data[all_vars]
        
        # Calculate correlation matrix
        corr_matrix = corr_data.corr()
        
        # Create comprehensive correlation visualization
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Stage 1: Comprehensive Correlation Analysis', fontsize=20, fontweight='bold')
        
        # 1. Full correlation heatmap (complete matrix)
        sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0,
                   square=True, fmt='.3f', cbar_kws={"shrink": .8}, ax=axes[0,0])
        axes[0,0].set_title('Full Correlation Matrix', fontsize=14, fontweight='bold')
        
        # 2. Feature-target correlations
        feature_target_corr = corr_matrix.loc[self.feature_names, self.target_names]
        sns.heatmap(feature_target_corr, annot=True, cmap='RdBu_r', center=0,
                   fmt='.3f', cbar_kws={"shrink": .8}, ax=axes[0,1])
        axes[0,1].set_title('Feature-Target Correlations', fontsize=14, fontweight='bold')
        
        # 3. Feature-feature correlations (collinearity check)
        feature_corr = corr_matrix.loc[self.feature_names, self.feature_names]
        sns.heatmap(feature_corr, annot=True, cmap='RdBu_r', center=0,
                   fmt='.3f', cbar_kws={"shrink": .8}, ax=axes[1,0])
        axes[1,0].set_title('Feature-Feature Correlations (Collinearity Check)', fontsize=14, fontweight='bold')
        
        # 4. Correlation strength distribution
        all_corr_values = []
        for i in range(len(all_vars)):
            for j in range(i+1, len(all_vars)):
                all_corr_values.append(abs(corr_matrix.iloc[i, j]))
        
        axes[1,1].hist(all_corr_values, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1,1].axvline(x=0.7, color='red', linestyle='--', linewidth=2, label='High Correlation (|r|=0.7)')
        axes[1,1].axvline(x=0.5, color='orange', linestyle='--', linewidth=2, label='Moderate Correlation (|r|=0.5)')
        axes[1,1].set_xlabel('Absolute Correlation Coefficient')
        axes[1,1].set_ylabel('Frequency')
        axes[1,1].set_title('Distribution of Correlation Strengths', fontsize=14, fontweight='bold')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Analyze and report findings
        print("\n📊 Correlation Analysis Results:")
        
        # High correlations
        high_corr_pairs = []
        for i in range(len(all_vars)):
            for j in range(i+1, len(all_vars)):
                corr_val = abs(corr_matrix.iloc[i, j])
                if corr_val > 0.7:
                    var1, var2 = all_vars[i], all_vars[j]
                    high_corr_pairs.append((var1, var2, corr_matrix.iloc[i, j]))
        
        if high_corr_pairs:
            print("   ⚠️  High correlations detected (|r| > 0.7):")
            for var1, var2, corr_val in high_corr_pairs:
                print(f"      • {var1} ↔ {var2}: r = {corr_val:.3f}")
        else:
            print("   ✅ No high correlations found (all |r| ≤ 0.7)")
        
        # Feature-target correlations summary
        print("\n   📈 Strongest feature-target correlations:")
        for target in self.target_names:
            target_corrs = [(feat, abs(feature_target_corr.loc[feat, target])) 
                           for feat in self.feature_names]
            target_corrs.sort(key=lambda x: x[1], reverse=True)
            print(f"      {target}:")
            for feat, corr_val in target_corrs:
                print(f"         {feat}: {feature_target_corr.loc[feat, target]:.3f}")
        
        # Store results
        self.results['stage1'] = {
            'correlation_matrix': corr_matrix,
            'high_correlations': high_corr_pairs,
            'feature_target_correlations': feature_target_corr
        }
        
        # VIF Analysis for collinearity
        self.calculate_vif()
        
        return corr_matrix
    
    def calculate_vif(self):
        """
        Calculate Variance Inflation Factor for collinearity assessment
        """
        print("\n🔍 Variance Inflation Factor (VIF) Analysis:")
        
        X = self.data[self.feature_names].values
        vif_data = pd.DataFrame()
        vif_data["Feature"] = self.feature_names
        vif_data["VIF"] = [variance_inflation_factor(X, i) for i in range(len(self.feature_names))]
        
        print("   VIF Values (>10 indicates high multicollinearity):")
        for idx, row in vif_data.iterrows():
            status = "⚠️" if row["VIF"] > 10 else ("🔶" if row["VIF"] > 5 else "✅")
            print(f"      {status} {row['Feature']}: {row['VIF']:.2f}")
        
        self.results['vif'] = vif_data

    def stage2_linear_baseline(self):
        """
        Stage 2: Linear Modeling Baseline - Multiple linear regression benchmark
        """
        print("\n" + "="*80)
        print("📏 STAGE 2: LINEAR MODELING BASELINE")
        print("="*80)
        
        if self.data is None:
            print("❌ Error: No data available")
            return
        
        linear_results = {}
        
        # Prepare features and targets
        X = self.data[self.feature_names]
        
        # Standardize features for linear regression
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=self.feature_names, index=X.index)
        
        print("🔄 Training linear regression models...")
        
        # Train models for each target
        for target in self.target_names:
            print(f"\n📊 Analyzing {target.replace('_', ' ')}:")
            
            y = self.data[target]
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=self.random_state
            )
            
            # Fit linear regression
            lr_model = LinearRegression()
            lr_model.fit(X_train, y_train)
            
            # Predictions
            y_pred_train = lr_model.predict(X_train)
            y_pred_test = lr_model.predict(X_test)
            
            # Metrics
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            
            # Cross-validation
            cv_scores = cross_val_score(lr_model, X_scaled, y, cv=5, scoring='r2')
            
            print(f"   • Training R²: {train_r2:.4f}")
            print(f"   • Test R²: {test_r2:.4f}")
            print(f"   • Training RMSE: {train_rmse:.2f}")
            print(f"   • Test RMSE: {test_rmse:.2f}")
            print(f"   • CV R² (mean±std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            
            # Feature coefficients
            coefficients = pd.DataFrame({
                'Feature': self.feature_names,
                'Coefficient': lr_model.coef_,
                'Abs_Coefficient': np.abs(lr_model.coef_)
            }).sort_values('Abs_Coefficient', ascending=False)
            
            print("   📈 Feature Coefficients (standardized):")
            for _, row in coefficients.iterrows():
                print(f"      {row['Feature']}: {row['Coefficient']:.4f}")
            
            # Store results
            linear_results[target] = {
                'model': lr_model,
                'scaler': scaler,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'cv_scores': cv_scores,
                'coefficients': coefficients,
                'y_pred_train': y_pred_train,
                'y_pred_test': y_pred_test,
                'y_train': y_train,
                'y_test': y_test
            }
        
        self.models['linear'] = linear_results
        self.results['stage2'] = linear_results
        
        # Create visualization
        self.visualize_linear_results()
        
        return linear_results
    
    def visualize_linear_results(self):
        """
        Visualize linear regression results
        """
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Stage 2: Linear Regression Analysis Results', fontsize=18, fontweight='bold')
        
        colors = ['#2E8B57', '#4169E1', '#DC143C']  # Sea Green, Royal Blue, Crimson
        
        # Top row: Predicted vs Actual
        for i, target in enumerate(self.target_names):
            ax = axes[0, i]
            results = self.results['stage2'][target]
            
            # Training points
            ax.scatter(results['y_train'], results['y_pred_train'], 
                      alpha=0.6, s=30, color=colors[i], label='Training', edgecolors='white', linewidth=0.5)
            
            # Test points  
            ax.scatter(results['y_test'], results['y_pred_test'], 
                      alpha=0.8, s=40, color='red', marker='^', label='Test', edgecolors='white', linewidth=0.5)
            
            # Perfect prediction line
            all_y = np.concatenate([results['y_train'], results['y_test']])
            all_pred = np.concatenate([results['y_pred_train'], results['y_pred_test']])
            min_val, max_val = min(all_y.min(), all_pred.min()), max(all_y.max(), all_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, linewidth=2)
            
            ax.set_xlabel('Actual Values')
            ax.set_ylabel('Predicted Values')
            ax.set_title(f'{target.replace("_", " ")}\nR² = {results["test_r2"]:.3f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Bottom row: Feature coefficients
        for i, target in enumerate(self.target_names):
            ax = axes[1, i]
            results = self.results['stage2'][target]
            coeffs = results['coefficients']
            
            bars = ax.barh(coeffs['Feature'], coeffs['Coefficient'], 
                          color=colors[i], alpha=0.7, edgecolor='white', linewidth=0.5)
            ax.set_xlabel('Standardized Coefficient')
            ax.set_title(f'{target.replace("_", " ")} - Feature Importance')
            ax.grid(True, alpha=0.3)
            ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
            
            # Add value labels
            for bar in bars:
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2, 
                       f'{width:.3f}', ha='left' if width >= 0 else 'right', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.show()

    def stage3_advanced_modeling(self):
        """
        Stage 3: Enhanced Linear Regression Analysis
        """
        print("\n" + "="*80)
        print("🚀 STAGE 3: ENHANCED LINEAR REGRESSION ANALYSIS")
        print("="*80)
        
        if self.data is None:
            print("❌ Error: No data available")
            return
        
        advanced_results = {}
        
        # Prepare features and targets
        X = self.data[self.feature_names]
        
        print("🔄 Enhanced Linear Regression Analysis...")
        
        # Train linear models for each target with more detailed analysis
        for target in self.target_names:
            print(f"\n🎯 Analyzing {target.replace('_', ' ')}:")
            
            y = self.data[target]
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=self.random_state
            )
            
            # Standardize features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Linear Regression with detailed analysis
            print("   📊 Training Enhanced Linear Regression...")
            lr_model = LinearRegression()
            lr_model.fit(X_train_scaled, y_train)
            
            lr_train_pred = lr_model.predict(X_train_scaled)
            lr_test_pred = lr_model.predict(X_test_scaled)
            
            lr_train_r2 = r2_score(y_train, lr_train_pred)
            lr_test_r2 = r2_score(y_test, lr_test_pred)
            lr_train_rmse = np.sqrt(mean_squared_error(y_train, lr_train_pred))
            lr_test_rmse = np.sqrt(mean_squared_error(y_test, lr_test_pred))
            lr_train_mae = mean_absolute_error(y_train, lr_train_pred)
            lr_test_mae = mean_absolute_error(y_test, lr_test_pred)
            
            # Cross-validation
            X_scaled = scaler.fit_transform(X)
            lr_cv_model = LinearRegression()
            lr_cv_scores = cross_val_score(lr_cv_model, X_scaled, y, cv=5, scoring='r2')
            
            print(f"      • Training R²: {lr_train_r2:.4f}")
            print(f"      • Test R²: {lr_test_r2:.4f}")
            print(f"      • CV R² (mean±std): {lr_cv_scores.mean():.4f} ± {lr_cv_scores.std():.4f}")
            print(f"      • Test RMSE: {lr_test_rmse:.4f}")
            print(f"      • Test MAE: {lr_test_mae:.4f}")
            
            # Feature coefficients analysis
            feature_importance = np.abs(lr_model.coef_)
            feature_importance = feature_importance / np.sum(feature_importance)  # Normalize
            
            target_results = {
                'linear_regression': {
                    'model': lr_model,
                    'scaler': scaler,
                    'train_r2': lr_train_r2,
                    'test_r2': lr_test_r2,
                    'train_rmse': lr_train_rmse,
                    'test_rmse': lr_test_rmse,
                    'train_mae': lr_train_mae,
                    'test_mae': lr_test_mae,
                    'cv_scores': lr_cv_scores,
                    'train_pred': lr_train_pred,
                    'test_pred': lr_test_pred,
                    'coefficients': lr_model.coef_,
                    'feature_importance': feature_importance
                }
            }
            
            target_results['best_model'] = 'linear_regression'
            print(f"      🏆 Linear Regression (R² = {lr_test_r2:.4f})")
            
            # Store train/test splits for later use
            target_results['X_train'] = X_train_scaled
            target_results['X_test'] = X_test_scaled
            target_results['y_train'] = y_train
            target_results['y_test'] = y_test
            
            advanced_results[target] = target_results
        
        self.models['advanced'] = advanced_results
        self.results['stage3'] = advanced_results
        
        # Create comprehensive visualizations
        self.visualize_advanced_results()
        
        return advanced_results
    
    def visualize_advanced_results(self):
        """
        Visualize linear regression results with comprehensive analysis
        """
        fig, axes = plt.subplots(3, 3, figsize=(22, 18))
        fig.suptitle('Stage 3: Enhanced Linear Regression Analysis', fontsize=20, fontweight='bold')
        
        colors = ['#2E8B57', '#4169E1', '#DC143C']  # Sea Green, Royal Blue, Crimson
        linear_colors = ['#A8DADC', '#457B9D']  # Light blue, darker blue for linear models
        
        # Row 1: Model Performance Metrics
        performance_metrics = {}
        for target in self.target_names:
            performance_metrics[target] = {}
            
            # Linear baseline from stage 2
            linear_results = self.results['stage2'][target]
            performance_metrics[target]['Stage 2 Linear'] = linear_results['test_r2']
            
            # Enhanced linear from stage 3
            enhanced_results = self.results['stage3'][target]
            performance_metrics[target]['Enhanced Linear'] = enhanced_results['linear_regression']['test_r2']
        
        # Performance comparison bar chart
        for i, target in enumerate(self.target_names):
            ax = axes[0, i]
            metrics = performance_metrics[target]
            models = list(metrics.keys())
            scores = list(metrics.values())
            
            bars = ax.bar(models, scores, color=linear_colors,
                         alpha=0.8, edgecolor='white', linewidth=1)
            ax.set_ylabel('Test R²')
            ax.set_title(f'{target.replace("_", " ")}\nModel Performance Comparison')
            ax.set_ylim(0, max(scores) * 1.1)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, score in zip(bars, scores):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Row 2: Feature Importance Comparison
        for i, target in enumerate(self.target_names):
            ax = axes[1, i]
            advanced_results = self.results['stage3'][target]
            
            # Get best model's feature importance
            best_model_type = advanced_results['best_model']
            importance = advanced_results[best_model_type]['feature_importance']
            
            # Create feature importance DataFrame
            importance_df = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': importance
            }).sort_values('Importance', ascending=True)
            
            bars = ax.barh(importance_df['Feature'], importance_df['Importance'], 
                          color=colors[i], alpha=0.7, edgecolor='white', linewidth=0.5)
            ax.set_xlabel('Feature Importance')
            ax.set_title(f'{target.replace("_", " ")}\nFeature Importance ({best_model_type.replace("_", " ").title()})')
            ax.grid(True, alpha=0.3, axis='x')
            
            # Add value labels
            for bar in bars:
                width = bar.get_width()
                ax.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                       f'{width:.3f}', ha='left', va='center', fontsize=9)
        
        # Row 3: Predicted vs Actual for Best Models
        for i, target in enumerate(self.target_names):
            ax = axes[2, i]
            advanced_results = self.results['stage3'][target]
            best_model_type = advanced_results['best_model']
            
            # Training points
            y_train = advanced_results['y_train']
            train_pred = advanced_results[best_model_type]['train_pred']
            ax.scatter(y_train, train_pred, alpha=0.6, s=30, color=colors[i], 
                      label='Training', edgecolors='white', linewidth=0.5)
            
            # Test points  
            y_test = advanced_results['y_test']
            test_pred = advanced_results[best_model_type]['test_pred']
            ax.scatter(y_test, test_pred, alpha=0.8, s=40, color='red', marker='^', 
                      label='Test', edgecolors='white', linewidth=0.5)
            
            # Perfect prediction line
            all_y = np.concatenate([y_train, y_test])
            all_pred = np.concatenate([train_pred, test_pred])
            min_val, max_val = min(all_y.min(), all_pred.min()), max(all_y.max(), all_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, linewidth=2)
            
            test_r2 = advanced_results[best_model_type]['test_r2']
            ax.set_xlabel('Actual Values')
            ax.set_ylabel('Predicted Values')
            ax.set_title(f'{target.replace("_", " ")}\n{best_model_type.replace("_", " ").title()} - R² = {test_r2:.3f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def stage4_deep_interpretation(self):
        """
        Stage 4: Linear Model Interpretation - Coefficient analysis and feature relationships
        """
        print("\n" + "="*80)
        print("🔬 STAGE 4: LINEAR MODEL INTERPRETATION")
        print("="*80)
        
        if self.data is None or 'stage3' not in self.results:
            print("❌ Error: Previous stages must be completed first")
            return
        
        interpretation_results = {}
        
        for target in self.target_names:
            print(f"\n🎯 Linear model interpretation for {target.replace('_', ' ')}:")
            
            target_results = self.results['stage3'][target]
            linear_model = target_results['linear_regression']['model']
            scaler = target_results['linear_regression']['scaler']
            
            target_interpretation = {}
            
            # 1. Coefficient Analysis
            print("   📊 Analyzing linear coefficients...")
            target_interpretation['coefficients'] = self.generate_coefficient_analysis(linear_model, scaler, target)
            
            # 2. Feature Relationship Analysis
            print("   📈 Analyzing feature relationships...")
            target_interpretation['relationships'] = self.analyze_feature_relationships(target)
            
            interpretation_results[target] = target_interpretation
        
        self.results['stage4'] = interpretation_results
        
        return interpretation_results
    
    def generate_coefficient_analysis(self, model, scaler, target):
        """
        Generate comprehensive linear coefficient analysis
        """
        # Get coefficients and feature names
        coefficients = model.coef_
        feature_names = self.feature_names
        
        # Create coefficient DataFrame
        coef_df = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': coefficients,
            'Abs_Coefficient': np.abs(coefficients)
        }).sort_values('Abs_Coefficient', ascending=True)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle(f'Linear Coefficient Analysis: {target.replace("_", " ")}', fontsize=16, fontweight='bold')
        
        # Plot 1: Coefficient values (with direction)
        colors = ['red' if x < 0 else 'blue' for x in coef_df['Coefficient']]
        bars1 = ax1.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors, alpha=0.7)
        ax1.set_xlabel('Coefficient Value')
        ax1.set_title('Coefficient Direction & Magnitude')
        ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, coef in zip(bars1, coef_df['Coefficient']):
            width = bar.get_width()
            label_x = width + (0.01 * max(np.abs(coef_df['Coefficient']))) if width >= 0 else width - (0.01 * max(np.abs(coef_df['Coefficient'])))
            ax1.text(label_x, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', ha='left' if width >= 0 else 'right', va='center', fontsize=9)
        
        # Plot 2: Feature importance (absolute coefficients)
        bars2 = ax2.barh(coef_df['Feature'], coef_df['Abs_Coefficient'], color='green', alpha=0.7)
        ax2.set_xlabel('Absolute Coefficient Value')
        ax2.set_title('Feature Importance (Magnitude)')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, coef in zip(bars2, coef_df['Abs_Coefficient']):
            width = bar.get_width()
            ax2.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.show()
        
        # Print coefficient interpretation
        print(f"   📋 Coefficient Summary:")
        print(f"      • Intercept: {model.intercept_:.4f}")
        print(f"      • Most influential features:")
        for _, row in coef_df.tail(3).iterrows():
            direction = "increases" if row['Coefficient'] > 0 else "decreases"
            print(f"         - {row['Feature']}: {direction} target by {row['Abs_Coefficient']:.4f} per unit")
        
        return {
            'coefficients': coefficients,
            'feature_names': feature_names,
            'intercept': model.intercept_,
            'coef_df': coef_df
        }
    
    def analyze_feature_relationships(self, target):
        """
        Analyze linear relationships between features and target
        """
        # Calculate correlations with target
        correlations = {}
        for feature in self.feature_names:
            if feature in self.data.columns:
                corr, p_value = pearsonr(self.data[feature], self.data[target])
                correlations[feature] = {'correlation': corr, 'p_value': p_value}
        
        # Create correlation plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        features = list(correlations.keys())
        corr_values = [correlations[f]['correlation'] for f in features]
        p_values = [correlations[f]['p_value'] for f in features]
        
        # Sort by absolute correlation
        sorted_indices = np.argsort(np.abs(corr_values))
        features = [features[i] for i in sorted_indices]
        corr_values = [corr_values[i] for i in sorted_indices]
        p_values = [p_values[i] for i in sorted_indices]
        
        # Color by significance
        colors = ['red' if p > 0.05 else ('orange' if p > 0.01 else 'green') for p in p_values]
        
        bars = ax.barh(features, corr_values, color=colors, alpha=0.7)
        ax.set_xlabel('Correlation with Target')
        ax.set_title(f'Feature-Target Correlations: {target.replace("_", " ")}')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax.grid(True, alpha=0.3)
        
        # Add value labels and significance
        for bar, corr, p_val in zip(bars, corr_values, p_values):
            width = bar.get_width()
            significance = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else ""))
            label_x = width + 0.01 if width >= 0 else width - 0.01
            ax.text(label_x, bar.get_y() + bar.get_height()/2, 
                   f'{width:.3f}{significance}', ha='left' if width >= 0 else 'right', va='center', fontsize=9)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', alpha=0.7, label='p < 0.01 (highly significant)'),
            Patch(facecolor='orange', alpha=0.7, label='0.01 ≤ p < 0.05 (significant)'), 
            Patch(facecolor='red', alpha=0.7, label='p ≥ 0.05 (not significant)')
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        plt.show()
        
        return correlations
    
    # SHAP analysis function removed as per requirements
    
    def advanced_correlation_analysis(self):
        """
        Advanced correlation analysis to handle potential masking effects
        """
        print("\n" + "="*80)
        print("🔬 ADVANCED CORRELATION ANALYSIS")
        print("Addressing potential masking of disaster effects by renewable penetration")
        print("="*80)
        
        # Stratified analysis by renewable penetration levels
        self.stratified_analysis()
        
        # Partial correlation analysis
        self.partial_correlation_analysis()
        
        # Residual analysis
        self.residual_analysis()
        
        # Interaction effects analysis
        self.interaction_effects_analysis()
        
        # Create comprehensive correlation visualization across all targets
        self.plot_comprehensive_correlation_analysis()
    
    def stratified_analysis(self):
        """
        Stratified analysis by renewable penetration levels
        """
        print("\n🎯 STRATIFIED ANALYSIS BY RENEWABLE PENETRATION")
        
        # Define renewable penetration groups
        renewable_col = 'Renewable_Penetration'
        data = self.data.copy()
        
        # Create quartile groups
        renewable_quartiles = data[renewable_col].quantile([0.25, 0.5, 0.75])
        
        # Define groups
        groups = {
            'Low (Q1)': data[data[renewable_col] <= renewable_quartiles[0.25]],
            'Med-Low (Q2)': data[(data[renewable_col] > renewable_quartiles[0.25]) & 
                                (data[renewable_col] <= renewable_quartiles[0.5])],
            'Med-High (Q3)': data[(data[renewable_col] > renewable_quartiles[0.5]) & 
                                 (data[renewable_col] <= renewable_quartiles[0.75])],
            'High (Q4)': data[data[renewable_col] > renewable_quartiles[0.75]]
        }
        
        print(f"   Renewable penetration quartiles:")
        print(f"      Q1 (≤{renewable_quartiles[0.25]:.3f}): {len(groups['Low (Q1)'])} samples")
        print(f"      Q2 ({renewable_quartiles[0.25]:.3f}-{renewable_quartiles[0.5]:.3f}]: {len(groups['Med-Low (Q2)'])} samples")
        print(f"      Q3 ({renewable_quartiles[0.5]:.3f}-{renewable_quartiles[0.75]:.3f}]: {len(groups['Med-High (Q3)'])} samples")
        print(f"      Q4 (>{renewable_quartiles[0.75]:.3f}): {len(groups['High (Q4)'])} samples")
        
        # Disaster features for analysis
        disaster_features = ['PDI', 'Max_Failure_Duration_Hours']
        
        # Analysis for each target
        stratified_results = {}
        
        for target in self.target_names:
            print(f"\n   📊 {target.replace('_', ' ')} Analysis by Renewable Penetration Groups:")
            target_results = {}
            
            for group_name, group_data in groups.items():
                if len(group_data) < 10:  # Skip groups with too few samples
                    continue
                    
                correlations = {}
                for feature in disaster_features:
                    if feature in group_data.columns and target in group_data.columns:
                        corr, p_value = pearsonr(group_data[feature], group_data[target])
                        correlations[feature] = {'correlation': corr, 'p_value': p_value}
                
                target_results[group_name] = correlations
                
                print(f"      {group_name}:")
                for feature, stats in correlations.items():
                    significance = "***" if stats['p_value'] < 0.001 else ("**" if stats['p_value'] < 0.01 else ("*" if stats['p_value'] < 0.05 else ""))
                    print(f"         {feature}: r = {stats['correlation']:.3f} (p = {stats['p_value']:.3f}){significance}")
            
            stratified_results[target] = target_results
        
        self.results['stratified_analysis'] = stratified_results
        
        # Create visualization
        self.plot_stratified_correlations(stratified_results, disaster_features)
    
    def plot_stratified_correlations(self, results, disaster_features):
        """
        Visualize stratified correlation results
        """
        fig, axes = plt.subplots(1, len(self.target_names), figsize=(6*len(self.target_names), 8))
        if len(self.target_names) == 1:
            axes = [axes]
        
        for i, target in enumerate(self.target_names):
            if target not in results:
                continue
                
            # Prepare data for heatmap
            groups = list(results[target].keys())
            features = disaster_features
            
            corr_matrix = np.zeros((len(features), len(groups)))
            
            for j, feature in enumerate(features):
                for k, group in enumerate(groups):
                    if feature in results[target][group]:
                        corr_matrix[j, k] = results[target][group][feature]['correlation']
                    else:
                        corr_matrix[j, k] = np.nan
            
            # Create heatmap
            im = axes[i].imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
            
            # Set labels
            axes[i].set_xticks(range(len(groups)))
            axes[i].set_xticklabels(groups, rotation=45)
            axes[i].set_yticks(range(len(features)))
            axes[i].set_yticklabels([f.replace('_', ' ') for f in features])
            axes[i].set_title(f'{target.replace("_", " ")}\nDisaster-Cost Correlations by Renewable Groups')
            
            # Add correlation values as text
            for j in range(len(features)):
                for k in range(len(groups)):
                    if not np.isnan(corr_matrix[j, k]):
                        text = axes[i].text(k, j, f'{corr_matrix[j, k]:.3f}',
                                          ha="center", va="center", color='black', fontsize=8)
        
        plt.colorbar(im, ax=axes, shrink=0.8, label='Correlation Coefficient')
        plt.tight_layout()
        plt.show()
        
        # Create additional correlation trend plots
        self.plot_correlation_trends(results, disaster_features)
    
    def plot_correlation_trends(self, stratified_results, disaster_features):
        """
        Plot correlation trends across renewable penetration levels
        """
        print("\n📈 Creating correlation trend visualizations...")
        
        # Create sliding window analysis for smoother trends
        self.plot_sliding_window_correlations(disaster_features)
        
        # Create detailed correlation comparison plots
        self.plot_detailed_correlation_comparison(stratified_results, disaster_features)
    
    def plot_sliding_window_correlations(self, disaster_features, window_size=0.15):
        """
        Plot correlations using sliding windows across renewable penetration spectrum
        """
        renewable_col = 'Renewable_Penetration'
        data = self.data.copy()
        
        # Define sliding windows
        renewable_min = data[renewable_col].min()
        renewable_max = data[renewable_col].max()
        renewable_range = renewable_max - renewable_min
        
        # Create window centers
        step_size = 0.05  # 5% steps
        window_centers = np.arange(renewable_min + window_size/2, 
                                 renewable_max - window_size/2 + step_size, 
                                 step_size)
        
        # Focus on LNG_Cost
        if 'LNG_Cost' not in self.target_names:
            return
        
        target = 'LNG_Cost'
        
        # Calculate correlations for each window
        correlation_trends = {}
        valid_centers = []
        
        for center in window_centers:
            window_min = center - window_size/2
            window_max = center + window_size/2
            
            # Get data in this window
            window_mask = (data[renewable_col] >= window_min) & (data[renewable_col] <= window_max)
            window_data = data[window_mask]
            
            if len(window_data) < 20:  # Need minimum samples
                continue
            
            valid_centers.append(center)
            
            # Calculate correlations for all disaster features
            for feature in disaster_features:
                if feature in window_data.columns:
                    corr, p_value = pearsonr(window_data[feature], window_data[target])
                    
                    if feature not in correlation_trends:
                        correlation_trends[feature] = {'correlations': [], 'p_values': []}
                    
                    correlation_trends[feature]['correlations'].append(corr)
                    correlation_trends[feature]['p_values'].append(p_value)
        
        if not correlation_trends:
            print("   ⚠️  Insufficient data for sliding window analysis")
            return
        
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        fig.suptitle(f'LNG Cost Correlations Across Renewable Penetration Spectrum\n(Sliding Window Analysis)', 
                    fontsize=16, fontweight='bold')
        
        # Plot 1: Correlation trends
        colors = plt.cm.Set1(np.linspace(0, 1, len(correlation_trends)))
        
        for i, (feature, trends) in enumerate(correlation_trends.items()):
            correlations = trends['correlations']
            p_values = trends['p_values']
            
            # Main correlation line
            ax1.plot(valid_centers, correlations, 'o-', color=colors[i], 
                    label=feature.replace('_', ' '), linewidth=2, markersize=4, alpha=0.8)
            
            # Add significance markers
            for j, (center, corr, p_val) in enumerate(zip(valid_centers, correlations, p_values)):
                if p_val < 0.05:
                    ax1.scatter(center, corr, s=50, color=colors[i], 
                              marker='*', edgecolors='black', linewidth=0.5)
        
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Renewable Penetration Level')
        ax1.set_ylabel('Correlation with LNG Cost')
        ax1.set_title('Disaster Indicators vs LNG Cost Correlations\n(Stars indicate p < 0.05)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Sample sizes in each window
        sample_sizes = []
        for center in valid_centers:
            window_min = center - window_size/2
            window_max = center + window_size/2
            window_mask = (data[renewable_col] >= window_min) & (data[renewable_col] <= window_max)
            sample_sizes.append(window_mask.sum())
        
        ax2.bar(valid_centers, sample_sizes, width=step_size*0.8, alpha=0.7, color='lightblue', edgecolor='navy')
        ax2.set_xlabel('Renewable Penetration Level')
        ax2.set_ylabel('Sample Size')
        ax2.set_title(f'Sample Sizes per Window (Window size: {window_size:.1%})')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print summary
        print(f"   📊 Analyzed {len(valid_centers)} windows across renewable penetration spectrum")
        print(f"   📊 Window size: {window_size:.1%} of renewable penetration range")
        
        # Identify features with strongest trend changes
        trend_changes = {}
        for feature, trends in correlation_trends.items():
            correlations = np.array(trends['correlations'])
            if len(correlations) > 3:
                # Calculate trend change (difference between first and last quartiles)
                q1_idx = len(correlations) // 4
                q3_idx = 3 * len(correlations) // 4
                trend_change = np.mean(correlations[q3_idx:]) - np.mean(correlations[:q1_idx])
                trend_changes[feature] = trend_change
        
        if trend_changes:
            print(f"\n   📈 Correlation trend changes (high vs low renewable penetration):")
            sorted_trends = sorted(trend_changes.items(), key=lambda x: abs(x[1]), reverse=True)
            for feature, change in sorted_trends:
                direction = "stronger" if change > 0 else "weaker"
                print(f"      {feature}: {change:+.3f} ({direction} correlation at high renewable penetration)")
    
    def plot_detailed_correlation_comparison(self, stratified_results, disaster_features):
        """
        Create detailed comparison plots for each disaster feature
        """
        if 'LNG_Cost' not in stratified_results:
            return
        
        lng_results = stratified_results['LNG_Cost']
        groups = list(lng_results.keys())
        
        # Create subplot for each disaster feature
        n_features = len(disaster_features)
        n_cols = 2
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        fig.suptitle('LNG Cost Correlations by Renewable Penetration Groups\n(Detailed Feature Analysis)', 
                    fontsize=16, fontweight='bold')
        
        for i, feature in enumerate(disaster_features):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]
            
            # Extract correlations and p-values for this feature
            correlations = []
            p_values = []
            sample_sizes = []
            
            for group in groups:
                if feature in lng_results[group]:
                    corr = lng_results[group][feature]['correlation']
                    p_val = lng_results[group][feature]['p_value']
                    correlations.append(corr)
                    p_values.append(p_val)
                    
                    # Get sample size for this group
                    renewable_col = 'Renewable_Penetration'
                    if group == 'Low (Q1)':
                        sample_size = (self.data[renewable_col] <= self.data[renewable_col].quantile(0.25)).sum()
                    elif group == 'Med-Low (Q2)':
                        q25, q50 = self.data[renewable_col].quantile([0.25, 0.5])
                        sample_size = ((self.data[renewable_col] > q25) & (self.data[renewable_col] <= q50)).sum()
                    elif group == 'Med-High (Q3)':
                        q50, q75 = self.data[renewable_col].quantile([0.5, 0.75])
                        sample_size = ((self.data[renewable_col] > q50) & (self.data[renewable_col] <= q75)).sum()
                    else:  # High (Q4)
                        sample_size = (self.data[renewable_col] > self.data[renewable_col].quantile(0.75)).sum()
                    
                    sample_sizes.append(sample_size)
                else:
                    correlations.append(np.nan)
                    p_values.append(np.nan)
                    sample_sizes.append(0)
            
            # Create bar plot with error indication
            colors = ['lightcoral' if p > 0.05 else 'steelblue' for p in p_values]
            bars = ax.bar(range(len(groups)), correlations, color=colors, alpha=0.8, edgecolor='black')
            
            # Add significance stars
            for j, (bar, p_val) in enumerate(zip(bars, p_values)):
                if not np.isnan(p_val):
                    height = bar.get_height()
                    if p_val < 0.001:
                        star = '***'
                    elif p_val < 0.01:
                        star = '**'
                    elif p_val < 0.05:
                        star = '*'
                    else:
                        star = ''
                    
                    if star:
                        ax.text(j, height + 0.02 if height >= 0 else height - 0.05, star, 
                               ha='center', va='bottom' if height >= 0 else 'top', 
                               fontweight='bold', fontsize=12)
            
            # Add correlation values on bars
            for j, (bar, corr, sample_size) in enumerate(zip(bars, correlations, sample_sizes)):
                if not np.isnan(corr):
                    height = bar.get_height()
                    ax.text(j, height/2, f'{corr:.3f}\n(n={sample_size})', 
                           ha='center', va='center', fontweight='bold', 
                           color='white' if abs(height) > 0.3 else 'black', fontsize=9)
            
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax.set_xticks(range(len(groups)))
            ax.set_xticklabels(groups, rotation=45)
            ax.set_ylabel('Correlation with LNG Cost')
            ax.set_title(f'{feature.replace("_", " ")}\nvs LNG Cost')
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim(-1, 1)
        
        # Remove empty subplots
        total_plots = n_rows * n_cols
        for i in range(n_features, total_plots):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].remove()
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='steelblue', alpha=0.8, label='Significant (p < 0.05)'),
            Patch(facecolor='lightcoral', alpha=0.8, label='Not significant (p ≥ 0.05)')
        ]
        fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
        
        plt.tight_layout()
        plt.show()
        
        print(f"   📊 Created detailed correlation comparison for {len(disaster_features)} disaster features")
        print(f"   📊 Stars indicate significance: *** p<0.001, ** p<0.01, * p<0.05")
    
    def partial_correlation_analysis(self):
        """
        Partial correlation analysis controlling for heating and cooling demand
        """
        print("\n🎯 PARTIAL CORRELATION ANALYSIS")
        print("   Controlling for Heating and Cooling Demand effects")
        
        from scipy.stats import pearsonr
        
        # Features to analyze (disaster intensity indicators)
        disaster_features = self.feature_names  # ['PDI', 'Max_Failure_Duration_Hours']
        control_vars = self.control_variables   # ['Heating_Demand', 'Cooling_Demand']
        
        partial_results = {}
        
        for target in self.target_names:
            print(f"\n   📊 {target.replace('_', ' ')}:")
            target_results = {}
            
            for feature in disaster_features:
                # Check if all required columns exist
                required_cols = [feature, target] + control_vars
                if all(col in self.data.columns for col in required_cols):
                    # Calculate partial correlation controlling for heating and cooling demand
                    partial_corr = self.calculate_partial_correlation_multiple(
                        self.data[feature], self.data[target], self.data[control_vars]
                    )
                    
                    # Regular correlation for comparison
                    regular_corr, _ = pearsonr(self.data[feature], self.data[target])
                    
                    target_results[feature] = {
                        'partial_correlation': partial_corr,
                        'regular_correlation': regular_corr,
                        'difference': partial_corr - regular_corr
                    }
                    
                    print(f"      {feature}:")
                    print(f"         Regular corr: {regular_corr:.3f}")
                    print(f"         Partial corr: {partial_corr:.3f} (controlling for {', '.join(control_vars)})")
                    print(f"         Difference:   {partial_corr - regular_corr:+.3f}")
            
            partial_results[target] = target_results
        
        self.results['partial_correlation'] = partial_results
    
    def calculate_partial_correlation_multiple(self, x, y, z_vars):
        """
        Calculate partial correlation between x and y, controlling for multiple variables in z_vars
        """
        import pandas as pd
        from scipy.stats import pearsonr
        
        # Combine all data
        data = pd.DataFrame({
            'x': x,
            'y': y
        })
        
        # Add control variables
        for i, z_var in enumerate(z_vars):
            data[f'z{i}'] = z_var
        
        # Remove NaN values
        data = data.dropna()
        
        if len(data) < 10:  # Need minimum sample size
            return 0
        
        try:
            # Use linear regression to remove effects of control variables
            import statsmodels.api as sm
            
            # Control variable columns
            Z = data[[f'z{i}' for i in range(len(z_vars))]]
            Z = sm.add_constant(Z)  # Add intercept
            
            # Regress x on control variables and get residuals
            model_x = sm.OLS(data['x'], Z).fit()
            residuals_x = model_x.resid
            
            # Regress y on control variables and get residuals  
            model_y = sm.OLS(data['y'], Z).fit()
            residuals_y = model_y.resid
            
            # Correlation between residuals is the partial correlation
            partial_corr, _ = pearsonr(residuals_x, residuals_y)
            
            return partial_corr
            
        except Exception as e:
            print(f"Warning: Partial correlation calculation failed: {e}")
            # Fallback to simple correlation
            return pearsonr(data['x'], data['y'])[0]
    
    def calculate_partial_correlation(self, x, y, z):
        """
        Calculate partial correlation between x and y, controlling for z (single variable)
        """
        # Calculate correlations
        r_xy, _ = pearsonr(x, y)
        r_xz, _ = pearsonr(x, z)
        r_yz, _ = pearsonr(y, z)
        
        # Partial correlation formula
        numerator = r_xy - (r_xz * r_yz)
        denominator = np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
        
        if denominator == 0:
            return 0
        else:
            return numerator / denominator
    
    def residual_analysis(self):
        """
        Residual analysis - Remove renewable penetration effects first
        """
        print("\n🎯 RESIDUAL ANALYSIS")
        print("   Analyzing disaster effects after removing renewable penetration influence")
        
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score
        
        disaster_features = ['PDI', 'Max_Failure_Duration_Hours']
        control_var = 'Renewable_Penetration'
        
        residual_results = {}
        
        for target in self.target_names:
            print(f"\n   📊 {target.replace('_', ' ')}:")
            
            # Step 1: Predict target using only renewable penetration
            X_control = self.data[[control_var]].values
            y_target = self.data[target].values
            
            model_control = LinearRegression()
            model_control.fit(X_control, y_target)
            y_pred_control = model_control.predict(X_control)
            
            # Calculate residuals (unexplained variance after renewable penetration)
            residuals = y_target - y_pred_control
            r2_control = r2_score(y_target, y_pred_control)
            
            print(f"      R² explained by {control_var}: {r2_control:.3f}")
            print(f"      Remaining variance: {1-r2_control:.3f}")
            
            # Step 2: Correlate residuals with disaster features
            residual_correlations = {}
            for feature in disaster_features:
                if feature in self.data.columns:
                    corr, p_value = pearsonr(residuals, self.data[feature])
                    residual_correlations[feature] = {
                        'correlation': corr,
                        'p_value': p_value
                    }
                    
                    significance = "***" if p_value < 0.001 else ("**" if p_value < 0.01 else ("*" if p_value < 0.05 else ""))
                    print(f"      {feature} vs residuals: r = {corr:.3f} (p = {p_value:.3f}){significance}")
            
            residual_results[target] = {
                'r2_control': r2_control,
                'residuals': residuals,
                'residual_correlations': residual_correlations
            }
        
        self.results['residual_analysis'] = residual_results
    
    def interaction_effects_analysis(self):
        """
        Interaction effects analysis between disaster indicators and renewable penetration
        """
        print("\n🎯 INTERACTION EFFECTS ANALYSIS")
        print("   Testing for interaction between disaster indicators and renewable penetration")
        
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import r2_score
        
        disaster_features = ['PDI', 'Max_Failure_Duration_Hours']
        renewable_var = 'Renewable_Penetration'
        
        interaction_results = {}
        
        for target in self.target_names:
            print(f"\n   📊 {target.replace('_', ' ')}:")
            target_results = {}
            
            for feature in disaster_features:
                if feature in self.data.columns:
                    # Prepare data
                    scaler = StandardScaler()
                    
                    # Main effects
                    X_main = self.data[[feature, renewable_var]].values
                    X_main_scaled = scaler.fit_transform(X_main)
                    
                    # Interaction effect (standardized to avoid scale issues)
                    interaction_term = (self.data[feature] * self.data[renewable_var]).values.reshape(-1, 1)
                    interaction_scaled = StandardScaler().fit_transform(interaction_term)
                    
                    # Combined model with interaction
                    X_interaction = np.concatenate([X_main_scaled, interaction_scaled], axis=1)
                    
                    y = self.data[target].values
                    
                    # Fit models
                    model_main = LinearRegression().fit(X_main_scaled, y)
                    model_interaction = LinearRegression().fit(X_interaction, y)
                    
                    # Calculate R² improvement
                    r2_main = r2_score(y, model_main.predict(X_main_scaled))
                    r2_interaction = r2_score(y, model_interaction.predict(X_interaction))
                    r2_improvement = r2_interaction - r2_main
                    
                    # Interaction coefficient
                    interaction_coef = model_interaction.coef_[-1]
                    
                    target_results[feature] = {
                        'r2_main': r2_main,
                        'r2_interaction': r2_interaction,
                        'r2_improvement': r2_improvement,
                        'interaction_coefficient': interaction_coef
                    }
                    
                    print(f"      {feature}:")
                    print(f"         Main effects R²: {r2_main:.3f}")
                    print(f"         With interaction R²: {r2_interaction:.3f}")
                    print(f"         R² improvement: {r2_improvement:+.3f}")
                    print(f"         Interaction coef: {interaction_coef:+.3f}")
                    
                    if r2_improvement > 0.01:  # Meaningful improvement threshold
                        print(f"         ⚠️  Significant interaction detected!")
            
            interaction_results[target] = target_results
        
        self.results['interaction_effects'] = interaction_results
    
    def plot_comprehensive_correlation_analysis(self):
        """
        Create comprehensive correlation analysis across all targets and renewable penetration levels
        """
        print("\n📊 COMPREHENSIVE CORRELATION VISUALIZATION")
        print("Creating correlation analysis across all target variables and renewable penetration levels...")
        
        # Create multi-target sliding window analysis
        self.plot_multi_target_correlation_trends()
        
        # Create correlation summary dashboard
        self.create_correlation_dashboard()
    
    def plot_multi_target_correlation_trends(self):
        """
        Plot correlation trends for all target variables
        """
        renewable_col = 'Renewable_Penetration'
        disaster_features = ['PDI', 'Max_Failure_Duration_Hours']
        data = self.data.copy()
        
        # Define sliding windows
        window_size = 0.15
        step_size = 0.08
        renewable_min = data[renewable_col].min()
        renewable_max = data[renewable_col].max()
        
        window_centers = np.arange(renewable_min + window_size/2, 
                                 renewable_max - window_size/2 + step_size, 
                                 step_size)
        
        # Create subplot for each target with Nature journal style
        fig, axes = plt.subplots(len(self.target_names), 1, figsize=(7, 2.5*len(self.target_names)))
        if len(self.target_names) == 1:
            axes = [axes]
        
        # Nature journal style colors - sophisticated and distinguishable
        nature_colors = ['#1f77b4', '#ff7f0e']  # Professional blue and orange for 2 features
        
        # Remove top and right spines for cleaner look
        plt.rcParams.update({
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.linewidth': 0.8,
            'xtick.major.width': 0.8,
            'ytick.major.width': 0.8,
            'font.size': 10
        })
        
        for target_idx, target in enumerate(self.target_names):
            ax = axes[target_idx]
            
            # Calculate correlations for each window
            for feature_idx, feature in enumerate(disaster_features):
                if feature not in data.columns:
                    continue
                
                correlations = []
                p_values = []
                valid_centers = []
                
                for center in window_centers:
                    window_min = center - window_size/2
                    window_max = center + window_size/2
                    
                    window_mask = (data[renewable_col] >= window_min) & (data[renewable_col] <= window_max)
                    window_data = data[window_mask]
                    
                    if len(window_data) < 15:
                        continue
                    
                    try:
                        corr, p_value = pearsonr(window_data[feature], window_data[target])
                        correlations.append(corr)
                        p_values.append(p_value)
                        valid_centers.append(center)
                    except:
                        continue
                
                if len(correlations) > 2:
                    # Plot correlation trend with Nature journal style
                    feature_label = 'Power dissipation index' if feature == 'PDI' else 'Maximum failure duration'
                    
                    # Main line with confidence band effect
                    ax.plot(valid_centers, correlations, '-', color=nature_colors[feature_idx], 
                           linewidth=2, alpha=0.9, label=feature_label)
                    
                    # Data points with subtle styling
                    ax.scatter(valid_centers, correlations, color=nature_colors[feature_idx], 
                             s=25, alpha=0.7, edgecolors='white', linewidth=0.5, zorder=3)
                    
                    # Significance markers with refined styling
                    significant_centers = [c for c, p in zip(valid_centers, p_values) if p < 0.05]
                    significant_corrs = [corr for corr, p in zip(correlations, p_values) if p < 0.05]
                    
                    if significant_centers:
                        ax.scatter(significant_centers, significant_corrs, s=35, 
                                 color=nature_colors[feature_idx], marker='s', 
                                 edgecolors='black', linewidth=0.8, zorder=4,
                                 alpha=0.9)
            
            # Nature journal style formatting
            ax.axhline(y=0, color='#666666', linestyle='-', alpha=0.4, linewidth=0.8)
            
            # Clean axis labels without bold (Nature style)
            ax.set_ylabel('Correlation coefficient', fontsize=10)
            
            # Simplified title
            target_label = target.replace('_', ' ').replace('Cost', 'cost').lower()
            if target_idx == 0:
                ax.set_title('Disaster impacts on energy system costs', fontsize=11, pad=15)
            
            # Nature style legend - inside plot area
            ax.legend(frameon=False, loc='upper right', fontsize=9)
            
            # Minimal grid
            ax.grid(True, alpha=0.2, linewidth=0.5)
            ax.set_ylim(-0.6, 0.6)
            
            # Remove top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(0.8)
            ax.spines['bottom'].set_linewidth(0.8)
            
            # Only show x-label on bottom subplot
            if target_idx == len(self.target_names) - 1:
                ax.set_xlabel('Renewable energy penetration', fontsize=10)
            
            # Add subplot labels (a, b, c) - Nature style
            subplot_labels = ['a', 'b', 'c']
            if target_idx < len(subplot_labels):
                ax.text(-0.1, 1.05, subplot_labels[target_idx], transform=ax.transAxes, 
                       fontsize=12, fontweight='bold', va='bottom', ha='right')
        
        # Add subtle confidence interval shading for better visual interpretation
        for target_idx, target in enumerate(self.target_names):
            ax = axes[target_idx]
            
            # Add y-axis ticks at meaningful intervals
            ax.set_yticks(np.arange(-0.6, 0.7, 0.2))
            ax.tick_params(axis='both', which='major', length=4, width=0.8)
            
            # Add subtle background color for negative correlation region
            ax.axhspan(-0.6, 0, alpha=0.05, color='red', zorder=0)
            ax.axhspan(0, 0.6, alpha=0.05, color='blue', zorder=0)
            
            # Add text annotations for correlation strength interpretation
            if target_idx == 0:
                ax.text(0.02, 0.95, 'Strong negative', transform=ax.transAxes, 
                       fontsize=8, alpha=0.7, ha='left', va='top')
                ax.text(0.02, 0.05, 'Strong positive', transform=ax.transAxes, 
                       fontsize=8, alpha=0.7, ha='left', va='bottom')
        
        # Adjust layout with Nature journal spacing
        plt.tight_layout(pad=1.5)
        plt.subplots_adjust(hspace=0.3)
        plt.show()
        
        print(f"   📈 Nature-style correlation analysis completed for {len(self.target_names)} cost variables")
        print(f"   🎨 Professional visualization with significance indicators (squares) applied")
    
    def create_correlation_dashboard(self):
        """
        Create a comprehensive correlation dashboard
        """
        disaster_features = ['PDI', 'Max_Failure_Duration_Hours']
        
        # Get stratified results if available
        if 'stratified_analysis' not in self.results:
            print("   ⚠️  Stratified analysis results not available for dashboard")
            return
        
        stratified_results = self.results['stratified_analysis']
        
        # Create dashboard with multiple subplots
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1, 1])
        
        fig.suptitle('Comprehensive Correlation Analysis Dashboard\nDisaster Indicators vs Cost Variables', 
                    fontsize=20, fontweight='bold')
        
        # Row 1: Correlation heatmaps for each target across renewable groups
        for target_idx, target in enumerate(self.target_names):
            if target not in stratified_results:
                continue
                
            ax = fig.add_subplot(gs[0, target_idx])
            
            groups = list(stratified_results[target].keys())
            features = [f for f in disaster_features if f in self.data.columns]
            
            # Create correlation matrix
            corr_matrix = np.zeros((len(features), len(groups)))
            
            for j, feature in enumerate(features):
                for k, group in enumerate(groups):
                    if feature in stratified_results[target][group]:
                        corr_matrix[j, k] = stratified_results[target][group][feature]['correlation']
                    else:
                        corr_matrix[j, k] = np.nan
            
            im = ax.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-0.8, vmax=0.8)
            
            # Add correlation values
            for j in range(len(features)):
                for k in range(len(groups)):
                    if not np.isnan(corr_matrix[j, k]):
                        ax.text(k, j, f'{corr_matrix[j, k]:.2f}', ha="center", va="center", 
                               color='white' if abs(corr_matrix[j, k]) > 0.4 else 'black', 
                               fontweight='bold', fontsize=8)
            
            ax.set_xticks(range(len(groups)))
            ax.set_xticklabels([g.split()[0] for g in groups], rotation=45)
            ax.set_yticks(range(len(features)))
            ax.set_yticklabels([f.replace('_', ' ')[:15] for f in features], fontsize=8)
            ax.set_title(f'{target.replace("_", " ")}', fontweight='bold', fontsize=12)
        
        # Add colorbar for heatmaps
        cbar_ax = fig.add_subplot(gs[0, 3])
        plt.colorbar(im, cax=cbar_ax, label='Correlation Coefficient')
        
        # Row 2: Correlation strength comparison
        ax_strength = fig.add_subplot(gs[1, :2])
        
        # Calculate average absolute correlations for each target
        avg_correlations = {}
        for target in self.target_names:
            if target in stratified_results:
                all_corrs = []
                for group_data in stratified_results[target].values():
                    for feature_data in group_data.values():
                        all_corrs.append(abs(feature_data['correlation']))
                avg_correlations[target] = np.mean(all_corrs) if all_corrs else 0
        
        targets = list(avg_correlations.keys())
        strengths = list(avg_correlations.values())
        
        bars = ax_strength.bar(targets, strengths, color=['steelblue', 'forestgreen', 'crimson'][:len(targets)], 
                              alpha=0.8, edgecolor='black')
        
        for bar, strength in zip(bars, strengths):
            ax_strength.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                           f'{strength:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax_strength.set_ylabel('Average |Correlation|', fontweight='bold')
        ax_strength.set_title('Average Correlation Strength\n(Disaster Indicators vs Cost Variables)', 
                             fontweight='bold')
        ax_strength.set_ylim(0, max(strengths) * 1.2 if strengths else 1)
        ax_strength.grid(True, alpha=0.3, axis='y')
        
        # Row 2-3: Renewable penetration distribution and masking analysis
        ax_dist = fig.add_subplot(gs[1, 2:])
        
        renewable_col = 'Renewable_Penetration'
        ax_dist.hist(self.data[renewable_col], bins=20, alpha=0.7, color='lightblue', 
                    edgecolor='navy', linewidth=1)
        ax_dist.axvline(self.data[renewable_col].mean(), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {self.data[renewable_col].mean():.3f}')
        ax_dist.axvline(self.data[renewable_col].median(), color='orange', linestyle='--', 
                       linewidth=2, label=f'Median: {self.data[renewable_col].median():.3f}')
        
        ax_dist.set_xlabel('Renewable Penetration Level', fontweight='bold')
        ax_dist.set_ylabel('Frequency', fontweight='bold')
        ax_dist.set_title('Renewable Penetration Distribution', fontweight='bold')
        ax_dist.legend()
        ax_dist.grid(True, alpha=0.3)
        
        # Row 3: Key insights text
        ax_insights = fig.add_subplot(gs[2, :])
        ax_insights.axis('off')
        
        insights_text = "KEY INSIGHTS FROM ADVANCED CORRELATION ANALYSIS:\n\n"
        
        # Add insights from results
        if 'partial_correlation' in self.results:
            insights_text += "• PARTIAL CORRELATION: Shows disaster effects while controlling for heating and cooling demand\n"
        
        if 'residual_analysis' in self.results:
            insights_text += "• RESIDUAL ANALYSIS: Analyzes disaster impacts on cost variance unexplained by renewables\n"
        
        if 'interaction_effects' in self.results:
            insights_text += "• INTERACTION EFFECTS: Tests if disaster impacts depend on renewable penetration levels\n"
        
        insights_text += "\n• MASKING EFFECT DETECTION: Strong renewable-cost correlations may hide disaster effects\n"
        insights_text += "• STRATIFIED ANALYSIS: Reveals how disaster-cost relationships vary across renewable levels\n"
        insights_text += "• SLIDING WINDOW: Provides smooth correlation trends across renewable penetration spectrum"
        
        ax_insights.text(0.05, 0.95, insights_text, transform=ax_insights.transAxes, 
                        fontsize=11, verticalalignment='top', fontweight='normal',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
        print("   📊 Comprehensive correlation dashboard created")
        print("   🔍 Dashboard includes: heatmaps, strength comparison, distribution, and key insights")

    def generate_comprehensive_report(self):
        """
        Generate a comprehensive analysis report
        """
        print("\n" + "="*80)
        print("📋 COMPREHENSIVE ANALYSIS REPORT")
        print("="*80)
        
        if not all(stage in self.results for stage in ['stage1', 'stage2', 'stage3']):
            print("❌ Error: Not all analysis stages completed")
            return
        
        print("\n🎯 SUMMARY OF FINDINGS:")
        print("-" * 40)
        
        # Linear regression performance comparison
        print("\n📊 LINEAR REGRESSION PERFORMANCE:")
        for target in self.target_names:
            print(f"\n   {target.replace('_', ' ').upper()}:")
            
            stage2_r2 = self.results['stage2'][target]['test_r2']
            stage3_results = self.results['stage3'][target]
            enhanced_r2 = stage3_results['linear_regression']['test_r2']
            
            print(f"      Stage 2 Linear:     R² = {stage2_r2:.4f}")
            print(f"      Enhanced Linear:    R² = {enhanced_r2:.4f}")
            
            improvement = enhanced_r2 - stage2_r2
            print(f"      📈 Enhancement:     {improvement:+.4f} ({improvement/stage2_r2*100:+.1f}%)")
        
        # Feature importance insights
        print("\n🔍 KEY FEATURE INSIGHTS:")
        for target in self.target_names:
            print(f"\n   {target.replace('_', ' ').upper()}:")
            
            advanced_results = self.results['stage3'][target]
            best_model_type = advanced_results['best_model']
            importance = advanced_results[best_model_type]['feature_importance']
            
            # Sort features by importance
            feature_importance = list(zip(self.feature_names, importance))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            print(f"      Most Important Features ({best_model_type.replace('_', ' ').title()}):")
            for feat, imp in feature_importance:
                print(f"         {feat.replace('_', ' ')}: {imp:.4f}")
        
        # Correlation insights
        print("\n📈 CORRELATION INSIGHTS:")
        high_corrs = self.results['stage1']['high_correlations']
        if high_corrs:
            print("   High correlations detected:")
            for var1, var2, corr in high_corrs:
                print(f"      {var1} ↔ {var2}: r = {corr:.3f}")
        else:
            print("   ✅ No problematic high correlations found")
        
        # VIF insights
        if 'vif' in self.results:
            vif_data = self.results['vif']
            high_vif = vif_data[vif_data['VIF'] > 5]
            if len(high_vif) > 0:
                print("\n   Multicollinearity concerns (VIF > 5):")
                for _, row in high_vif.iterrows():
                    print(f"      {row['Feature']}: VIF = {row['VIF']:.2f}")
            else:
                print("\n   ✅ No multicollinearity concerns (all VIF ≤ 5)")
        
        # Advanced correlation analysis insights
        if 'stratified_analysis' in self.results:
            print("\n🧪 ADVANCED CORRELATION INSIGHTS:")
            print("   Stratified Analysis Results:")
            # Print key findings from stratified analysis
            for target in self.target_names:
                if target in self.results['stratified_analysis']:
                    print(f"      {target.replace('_', ' ')}: Group-specific correlations analyzed")
        
        if 'partial_correlation' in self.results:
            print("   Partial Correlation Analysis:")
            print("      Disaster effects analyzed while controlling for heating and cooling demand")
        
        if 'residual_analysis' in self.results:
            print("   Residual Analysis:")
            print("      Disaster effects on cost variance unexplained by renewable penetration")
        
        if 'interaction_effects' in self.results:
            print("   Interaction Effects Analysis:")
            significant_interactions = []
            for target, features in self.results['interaction_effects'].items():
                for feature, stats in features.items():
                    if stats.get('r2_improvement', 0) > 0.01:
                        significant_interactions.append(f"{feature} × Renewable_Penetration → {target}")
            
            if significant_interactions:
                print("      ⚠️  Significant interactions detected:")
                for interaction in significant_interactions:
                    print(f"         {interaction}")
            else:
                print("      ✅ No significant interaction effects found")
        
        print("\n" + "="*80)
        print("✅ COMPREHENSIVE ANALYSIS COMPLETED!")
        print("="*80)


if __name__ == "__main__":
    # Initialize analysis
    analyzer = AdvancedEnergyAnalysis(random_state=42)
    
    print("Advanced Energy System Investment Analysis")
    print("Using progressive modeling approach:")
    print("   1. Correlation Analysis (Global View)")
    print("   2. Linear Baseline (Benchmark)")
    print("   3. Enhanced Linear Analysis")
    print("   4. Advanced Correlation Analysis")
    print()
    
    # Load and prepare data
    df = analyzer.calculate_real_data()
    if df is None:
        print("❌ Failed to load data. Please check file paths.")
        exit(1)
    
    # Apply data cleaning
    df_cleaned = analyzer.apply_data_cleaning(strategy='moderate')
    if len(df_cleaned) < 30:
        print("❌ Insufficient data after cleaning")
        exit(1)
    
    # Execute all analysis stages
    print("\n🎬 Starting comprehensive four-stage analysis...")
    
    try:
        # Stage 1: Correlation Analysis
        analyzer.stage1_correlation_analysis()
        
        # Stage 2: Linear Baseline
        analyzer.stage2_linear_baseline()
        
        # Stage 3: Enhanced Linear Analysis
        analyzer.stage3_advanced_modeling()
        
        # Stage 4: Linear Model Interpretation
        analyzer.stage4_deep_interpretation()
        
        # Stage 5: Advanced Correlation Analysis (addressing masking effects)
        analyzer.advanced_correlation_analysis()
        
        # Generate comprehensive report
        analyzer.generate_comprehensive_report()
        
        print("\n🎉 ALL ANALYSIS STAGES COMPLETED SUCCESSFULLY!")
        print("📊 Linear regression analysis has been applied with comprehensive interpretation.")
        print("🔍 Results include correlation analysis, linear modeling, feature importance, and coefficient analysis.")
        print("🧪 Advanced correlation analysis with partial correlation and stratified grouping completed.")
        
    except Exception as e:
        print(f"\n❌ Analysis failed with error: {str(e)}")
        print("🔧 Please check data availability and requirements.")
        import traceback
        traceback.print_exc()