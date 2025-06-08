"""
BigMart Sales Prediction 
==============================================

A comprehensive machine learning solution for predicting retail sales using ensemble methods.
This solution achieved Rank #250 out of 5000+ participants in the BigMart Sales Prediction competition.

Author: Bikram Kumar Das
Date: 6th June 2025
Competition: BigMart Sales Prediction Hackathon
Final Rank: #250 (RMSE: 1146.099877087)

Key Features:
- Advanced data preprocessing with domain knowledge integration
- Strategic feature engineering based on retail business logic
- Ensemble modeling with optimized hyperparameters
- Systematic approach to model validation and selection
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import warnings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    logger.warning("CatBoost not available. Install with: pip install catboost")

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class BigMartSalesPredictor:
    """
    A comprehensive machine learning solution for retail sales prediction.
    
    This class implements an ensemble approach using CatBoost models with different
    random seeds, incorporating domain-specific feature engineering and advanced
    preprocessing techniques optimized for retail data.
    
    Attributes:
        label_encoders (dict): Storage for categorical encoders
        models (dict): Trained CatBoost models for each seed
        ensemble_weights (dict): Calculated weights for ensemble averaging
        categorical_features (list): List of categorical feature names
        target_seeds (list): Random seeds for ensemble diversity
    """
    
    def __init__(self, target_seeds=[46, 48]):
        """
        Initialize the BigMart Sales Predictor.
        
        Args:
            target_seeds (list): Random seeds for creating diverse models.
                                Selected based on empirical validation performance.
        """
        self.label_encoders = {}
        self.models = {}
        self.ensemble_weights = {}
        self.train_df = None
        self.test_df = None
        self.categorical_features = []
        self.target_seeds = target_seeds
        
        logger.info(f"Initialized BigMart Sales Predictor with seeds: {target_seeds}")
    
    def load_data(self, train_path, test_path):
        """
        Load training and test datasets from CSV files.
        
        Args:
            train_path (str): Path to training dataset
            test_path (str): Path to test dataset
            
        Returns:
            tuple: (train_df, test_df) pandas DataFrames
        """
        try:
            self.train_df = pd.read_csv(train_path)
            self.test_df = pd.read_csv(test_path)
            
            logger.info(f"Training data loaded: {self.train_df.shape}")
            logger.info(f"Test data loaded: {self.test_df.shape}")
            
            # Basic data validation
            assert not self.train_df.empty, "Training data is empty"
            assert not self.test_df.empty, "Test data is empty"
            assert 'Item_Outlet_Sales' in self.train_df.columns, "Target variable missing"
            
            return self.train_df, self.test_df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def preprocess_data(self):
        """
        Comprehensive data preprocessing pipeline incorporating domain knowledge.
        
        This method implements several critical preprocessing steps:
        1. Intelligent missing value imputation using business logic
        2. Data standardization and cleaning
        3. Advanced feature engineering based on retail domain expertise
        4. Categorical encoding optimized for tree-based models
        
        Returns:
            tuple: (processed_train_df, processed_test_df)
        """
        logger.info("Starting comprehensive data preprocessing")
        
        # Combine datasets for consistent preprocessing
        # This ensures identical feature distributions across train/test splits
        test_df_copy = self.test_df.copy()
        test_df_copy['Item_Outlet_Sales'] = 0  # Placeholder for target variable
        combined_df = pd.concat([self.train_df, test_df_copy], ignore_index=True)
        
        logger.info(f"Combined dataset shape: {combined_df.shape}")
        
        # Step 1: Intelligent Missing Value Imputation
        self._handle_missing_values(combined_df)
        
        # Step 2: Data Standardization and Business Logic Corrections
        self._standardize_categorical_data(combined_df)
        
        # Step 3: Advanced Feature Engineering
        self._create_engineered_features(combined_df)
        
        # Step 4: Categorical Encoding for Machine Learning
        self._encode_categorical_features(combined_df)
        
        # Split back into train and test sets
        self.train_processed = combined_df[:len(self.train_df)].copy()
        self.test_processed = combined_df[len(self.train_df):].copy()
        self.test_processed.drop('Item_Outlet_Sales', axis=1, inplace=True)
        
        logger.info(f"Preprocessing completed - Train: {self.train_processed.shape}, Test: {self.test_processed.shape}")
        
        return self.train_processed, self.test_processed
    
    def _handle_missing_values(self, df):
        """
        Implement domain-specific missing value imputation strategies.
        
        Args:
            df (pandas.DataFrame): Combined dataset to process
            
        Business Logic Applied:
        - Item weights: Use item-specific averages (same items have similar weights)
        - Outlet sizes: Use most common size (mode imputation)
        - Zero visibility: Replace with item-specific averages (business constraint)
        """
        logger.info("Handling missing values with domain-specific strategies")
        
        # Item Weight: Use item-specific imputation
        # Rationale: Same items across outlets should have similar weights
        item_weight_avg = df.groupby('Item_Identifier')['Item_Weight'].mean()
        missing_weight_mask = df['Item_Weight'].isnull()
        df.loc[missing_weight_mask, 'Item_Weight'] = df.loc[missing_weight_mask, 'Item_Identifier'].map(item_weight_avg)
        
        # Fill any remaining missing weights with overall mean
        df['Item_Weight'].fillna(df['Item_Weight'].mean(), inplace=True)
        
        # Outlet Size: Use mode imputation
        # Rationale: Most outlets fall into common size categories
        df['Outlet_Size'].fillna(df['Outlet_Size'].mode()[0], inplace=True)
        
        logger.info("Missing value imputation completed")
    
    def _standardize_categorical_data(self, df):
        """
        Standardize categorical data and apply business logic corrections.
        
        Args:
            df (pandas.DataFrame): Dataset to standardize
            
        Business Logic Applied:
        - Standardize fat content labels (LF -> Low Fat, reg -> Regular)
        - Set non-edible items to have 'Non-Edible' fat content
        - Handle zero visibility cases (business constraint violation)
        """
        logger.info("Standardizing categorical data and applying business logic")
        
        # Standardize Item_Fat_Content inconsistencies
        # Business logic: Standardize various representations of the same concept
        standardization_map = {
            'LF': 'Low Fat',
            'reg': 'Regular', 
            'low fat': 'Low Fat'
        }
        df['Item_Fat_Content'] = df['Item_Fat_Content'].replace(standardization_map)
        
        # Apply business logic: Non-edible items cannot have fat content
        non_edible_categories = ['Health and Hygiene', 'Household', 'Others']
        non_edible_mask = df['Item_Type'].isin(non_edible_categories)
        df.loc[non_edible_mask, 'Item_Fat_Content'] = 'Non-Edible'
        
        # Handle zero visibility (business constraint)
        # Rationale: Zero visibility is unrealistic; use item-specific averages
        zero_visibility_mask = df['Item_Visibility'] == 0
        item_visibility_avg = df.groupby('Item_Identifier')['Item_Visibility'].mean()
        df.loc[zero_visibility_mask, 'Item_Visibility'] = df.loc[zero_visibility_mask, 'Item_Identifier'].map(item_visibility_avg)
        
        logger.info("Categorical data standardization completed")
    
    def _create_engineered_features(self, df):
        """
        Create advanced features based on retail domain expertise.
        
        Args:
            df (pandas.DataFrame): Dataset for feature engineering
            
        Feature Categories Created:
        1. Temporal features (outlet age, establishment patterns)
        2. Product categorization (food vs non-food, perishability)
        3. Economic features (price positioning, value metrics)
        4. Market structure features (item distribution, outlet diversity)
        5. Interaction features (cross-variable relationships)
        """
        logger.info("Creating advanced engineered features")
        
        # Temporal Features
        # Outlet age is crucial for understanding market maturity
        df['Outlet_Age'] = 2013 - df['Outlet_Establishment_Year']
        
        # Product Categorization Features
        # Extract product category from item identifier (business logic)
        df['Item_Type_Category'] = df['Item_Identifier'].str[:2]
        
        # Economic Features
        # Price per weight ratio indicates product value positioning
        df['Price_per_Weight'] = df['Item_MRP'] / df['Item_Weight']
        
        # Create price segments using business-informed bins
        df['MRP_Category'] = pd.cut(df['Item_MRP'], 
                                   bins=[0, 69, 136, 203, 270],
                                   labels=['Low', 'Medium', 'High', 'Very High'])
        
        # Create price quantiles for finer granularity
        df['MRP_Quantile'] = pd.qcut(df['Item_MRP'], q=5, 
                                    labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
        
        # Visibility categorization (shelf space importance)
        df['Visibility_Category'] = pd.cut(df['Item_Visibility'],
                                          bins=[0, 0.065, 0.13, 0.2],
                                          labels=['Low', 'Medium', 'High'])
        
        # Business Logic Categorization
        # Food vs Non-Food categorization based on domain knowledge
        food_categories = [
            'Baking Goods', 'Breads', 'Breakfast', 'Dairy', 
            'Fruits and Vegetables', 'Meat', 'Seafood', 
            'Snack Foods', 'Starchy Food'
        ]
        df['Food_Category'] = df['Item_Type'].apply(
            lambda x: 'Food' if x in food_categories else 'Non_Food'
        )
        
        # Perishability classification (affects inventory and sales patterns)
        perishable_categories = [
            'Breads', 'Breakfast', 'Dairy', 
            'Fruits and Vegetables', 'Meat', 'Seafood'
        ]
        df['Perishability'] = df['Item_Type'].apply(
            lambda x: 'Perishable' if x in perishable_categories else 'Non_Perishable'
        )
        
        # Outlet categorization (business model grouping)
        outlet_mapping = {
            'Grocery Store': 'Grocery',
            'Supermarket Type1': 'Supermarket',
            'Supermarket Type2': 'Supermarket', 
            'Supermarket Type3': 'Supermarket'
        }
        df['Outlet_Type_Category'] = df['Outlet_Type'].map(outlet_mapping)
        
        # Outlet maturity segmentation
        df['Outlet_Age_Group'] = pd.cut(df['Outlet_Age'],
                                       bins=[0, 8, 16, 30],
                                       labels=['New', 'Medium', 'Established'])
        
        # Market Structure Features
        # Price positioning within category (competitive analysis)
        df['Price_Rank_in_Category'] = df.groupby('Item_Type')['Item_MRP'].rank(pct=True)
        
        # Item market penetration (how widespread is this item)
        item_outlet_count = df.groupby('Item_Identifier')['Outlet_Identifier'].nunique()
        df['Item_Outlet_Count'] = df['Item_Identifier'].map(item_outlet_count)
        
        # Outlet product diversity (business model indicator)
        outlet_item_count = df.groupby('Outlet_Identifier')['Item_Identifier'].nunique()
        df['Outlet_Item_Diversity'] = df['Outlet_Identifier'].map(outlet_item_count)
        
        # Strategic Interaction Features
        # These capture complex business relationships
        df['Food_Outlet_Type'] = df['Food_Category'].astype(str) + '_' + df['Outlet_Type'].astype(str)
        df['Perishable_Outlet_Type'] = df['Perishability'].astype(str) + '_' + df['Outlet_Type'].astype(str)
        df['MRP_Quantile_Outlet'] = df['MRP_Quantile'].astype(str) + '_' + df['Outlet_Type'].astype(str)
        
        logger.info("Feature engineering completed - Created 17 additional features")
    
    def _encode_categorical_features(self, df):
        """
        Encode categorical features for machine learning compatibility.
        
        Args:
            df (pandas.DataFrame): Dataset with categorical features to encode
            
        Uses Label Encoding for all categorical features as CatBoost can handle
        categorical features natively, but encoded features often perform better
        in practice for this specific problem domain.
        """
        logger.info("Encoding categorical features")
        
        # Define all categorical features for encoding
        categorical_columns = [
            'Item_Fat_Content', 'Item_Type', 'Outlet_Identifier',
            'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type',
            'Item_Type_Category', 'MRP_Category', 'Visibility_Category',
            'Outlet_Type_Category', 'MRP_Quantile', 'Food_Category',
            'Perishability', 'Outlet_Age_Group', 'Food_Outlet_Type',
            'Perishable_Outlet_Type', 'MRP_Quantile_Outlet'
        ]
        
        # Store categorical feature names for CatBoost
        self.categorical_features = categorical_columns.copy()
        
        # Apply label encoding to each categorical feature
        for column in categorical_columns:
            if column in df.columns:
                encoder = LabelEncoder()
                df[column] = encoder.fit_transform(df[column].astype(str))
                self.label_encoders[column] = encoder
        
        logger.info(f"Encoded {len(categorical_columns)} categorical features")
    
    def prepare_features(self):
        """
        Prepare final feature set for model training.
        
        Returns:
            tuple: (X_train, y_train, X_test) prepared for machine learning
            
        Feature Selection Strategy:
        - Include all original features (domain completeness)
        - Include all engineered features (enhanced predictive power)
        - Total: 25 features optimized for retail sales prediction
        """
        logger.info("Preparing final feature set")
        
        # Define comprehensive feature set
        feature_columns = [
            # Original features
            'Item_Weight', 'Item_Fat_Content', 'Item_Visibility', 'Item_Type',
            'Item_MRP', 'Outlet_Identifier', 'Outlet_Establishment_Year',
            'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type',
            
            # Engineered features  
            'Outlet_Age', 'Item_Type_Category', 'Price_per_Weight',
            'MRP_Category', 'Visibility_Category', 'Outlet_Type_Category',
            'MRP_Quantile', 'Food_Category', 'Perishability', 'Outlet_Age_Group',
            'Price_Rank_in_Category', 'Item_Outlet_Count', 'Outlet_Item_Diversity',
            'Food_Outlet_Type', 'Perishable_Outlet_Type', 'MRP_Quantile_Outlet'
        ]
        
        # Select available features (safety check)
        available_features = [col for col in feature_columns if col in self.train_processed.columns]
        
        # Prepare training data
        self.X_train = self.train_processed[available_features]
        self.y_train = self.train_processed['Item_Outlet_Sales']
        self.X_test = self.test_processed[available_features]
        
        # Identify categorical feature indices for CatBoost
        self.cat_feature_indices = [
            i for i, col in enumerate(available_features) 
            if col in self.categorical_features
        ]
        
        logger.info(f"Feature preparation completed - {len(available_features)} features selected")
        logger.info(f"Categorical features: {len(self.cat_feature_indices)}")
        
        return self.X_train, self.y_train, self.X_test
    
    def train_ensemble_models(self):
        """
        Train ensemble of CatBoost models with optimized hyperparameters.
        
        Ensemble Strategy:
        - Multiple models with different random seeds for diversity
        - Performance-based weighting using validation scores
        - Micro-tuned hyperparameters based on systematic optimization
        
        Returns:
            tuple: (trained_models, ensemble_weights)
        """
        logger.info("Training ensemble of CatBoost models")
        
        validation_predictions = {}
        validation_scores = {}
        
        # Optimized hyperparameters based on systematic tuning
        optimized_params = {
            'iterations': 105,          # Micro-tuned: +5 from baseline
            'learning_rate': 0.095,     # Micro-tuned: -0.005 from baseline  
            'depth': 6,                 # Optimal depth for this problem
            'random_seed': None,        # Will be set per model
            'verbose': False,
            'thread_count': -1
        }
        
        for seed in self.target_seeds:
            logger.info(f"Training model with seed {seed}")
            
            # Set random seed for reproducibility
            np.random.seed(seed)
            
            # Create validation split
            X_train_fold, X_val_fold, y_train_fold, y_val_fold = train_test_split(
                self.X_train, self.y_train, test_size=0.2, random_state=seed
            )
            
            # Configure model with current seed
            model_params = optimized_params.copy()
            model_params['random_seed'] = seed
            model_params['cat_features'] = self.cat_feature_indices
            
            # Train CatBoost model
            model = cb.CatBoostRegressor(**model_params)
            model.fit(X_train_fold, y_train_fold)
            
            # Validate model performance
            val_predictions = model.predict(X_val_fold)
            val_rmse = np.sqrt(mean_squared_error(y_val_fold, val_predictions))
            
            # Store results
            self.models[f'seed_{seed}'] = model
            validation_predictions[f'seed_{seed}'] = val_predictions
            validation_scores[f'seed_{seed}'] = val_rmse
            
            logger.info(f"Seed {seed} - Validation RMSE: {val_rmse:.4f}")
            
            # Retrain on full training data for final predictions
            model.fit(self.X_train, self.y_train)
        
        # Calculate performance-based ensemble weights
        self._calculate_ensemble_weights(validation_scores)
        
        logger.info("Ensemble training completed")
        return self.models, self.ensemble_weights
    
    def _calculate_ensemble_weights(self, validation_scores):
        """
        Calculate performance-based weights for ensemble averaging.
        
        Args:
            validation_scores (dict): Validation RMSE scores for each model
            
        Weighting Strategy:
        - Inverse weighting: Lower RMSE = Higher weight
        - Normalized weights: Sum to 1.0 for proper averaging
        """
        logger.info("Calculating performance-based ensemble weights")
        
        # Convert RMSE scores to weights (inverse relationship)
        rmse_values = np.array(list(validation_scores.values()))
        inverse_weights = 1.0 / rmse_values
        normalized_weights = inverse_weights / inverse_weights.sum()
        
        # Store normalized weights
        self.ensemble_weights = {}
        for i, seed in enumerate(self.target_seeds):
            weight = normalized_weights[i]
            self.ensemble_weights[f'seed_{seed}'] = weight
            logger.info(f"Seed {seed}: Weight = {weight:.4f}, RMSE = {validation_scores[f'seed_{seed}']:.4f}")
    
    def generate_predictions(self):
        """
        Generate ensemble predictions using trained models.
        
        Returns:
            numpy.ndarray: Final ensemble predictions
            
        Ensemble Method:
        - Weighted average of individual model predictions
        - Weights based on validation performance
        - Post-processing to ensure business constraints (no negative sales)
        """
        logger.info("Generating ensemble predictions")
        
        individual_predictions = {}
        
        # Get predictions from each model
        for model_name, model in self.models.items():
            predictions = model.predict(self.X_test)
            predictions = np.maximum(predictions, 0)  # Business constraint: No negative sales
            individual_predictions[model_name] = predictions
            
            seed = model_name.split('_')[1]
            logger.info(f"Seed {seed} - Prediction range: [{predictions.min():.2f}, {predictions.max():.2f}]")
        
        # Create weighted ensemble prediction
        ensemble_predictions = np.zeros_like(list(individual_predictions.values())[0])
        for model_name, predictions in individual_predictions.items():
            weight = self.ensemble_weights[model_name]
            ensemble_predictions += weight * predictions
        
        logger.info(f"Ensemble predictions - Range: [{ensemble_predictions.min():.2f}, {ensemble_predictions.max():.2f}]")
        logger.info(f"Ensemble predictions - Mean: {ensemble_predictions.mean():.2f}")
        
        return ensemble_predictions
    
    def create_submission(self, predictions, filename='bigmart_ensemble_submission.csv'):
        """
        Create competition submission file.
        
        Args:
            predictions (numpy.ndarray): Model predictions
            filename (str): Output filename
            
        Returns:
            pandas.DataFrame: Submission dataframe
        """
        logger.info(f"Creating submission file: {filename}")
        
        submission = pd.DataFrame({
            'Item_Identifier': self.test_df['Item_Identifier'],
            'Outlet_Identifier': self.test_df['Outlet_Identifier'], 
            'Item_Outlet_Sales': predictions
        })
        
        # Save submission file
        submission.to_csv(filename, index=False)
        
        logger.info(f"Submission file created successfully")
        logger.info(f"Submission shape: {submission.shape}")
        logger.info(f"Sample predictions:\n{submission.head()}")
        
        return submission
    
    def run_complete_pipeline(self, train_path, test_path, submission_filename='final_submission.csv'):
        """
        Execute the complete machine learning pipeline.
        
        Args:
            train_path (str): Path to training data
            test_path (str): Path to test data  
            submission_filename (str): Output submission filename
            
        Returns:
            pandas.DataFrame: Final submission dataframe
        """
        logger.info("Starting complete BigMart Sales Prediction pipeline")
        
        try:
            # Step 1: Data Loading
            self.load_data(train_path, test_path)
            
            # Step 2: Data Preprocessing
            self.preprocess_data()
            
            # Step 3: Feature Preparation
            self.prepare_features()
            
            # Step 4: Model Training
            self.train_ensemble_models()
            
            # Step 5: Prediction Generation
            predictions = self.generate_predictions()
            
            # Step 6: Submission Creation
            submission = self.create_submission(predictions, submission_filename)
            
            logger.info("Pipeline completed successfully")
            logger.info(f"Final submission saved as: {submission_filename}")
            
            return submission
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise

def main():
    """
    Main execution function for the BigMart Sales Prediction solution.
    """
    # Initialize predictor with optimized seeds
    # Seeds 46 and 48 were selected through systematic validation
    predictor = BigMartSalesPredictor(target_seeds=[46, 48])
    
    # Execute complete pipeline
    submission = predictor.run_complete_pipeline(
        train_path='train_v9rqX0R.csv',
        test_path='test_AbJTz2l.csv',
        submission_filename='bigmart_final_submission.csv'
    )
    
    print("BigMart Sales Prediction - Solution Complete")
    print(f"Submission file: bigmart_final_submission.csv")
    print("Expected Performance: RMSE ~1146.10 (Rank #250)")

if __name__ == "__main__":
    main()