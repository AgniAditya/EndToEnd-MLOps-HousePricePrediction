# Data Cleaning and Preprocessing Module
# This module handles all data preprocessing operations including cleaning, encoding, and splitting
# It follows the Strategy pattern with abstract DataStrategy class and concrete implementations
from abc import ABC
from typing import Union
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple
from typing_extensions import Annotated


class DataStrategy(ABC):
    """
    Abstract base class defining the strategy pattern for data handling operations
    
    This class provides a common interface for different data processing strategies.
    All concrete data processing classes must inherit from this class and implement
    the handle_data method. This allows for easy swapping of different preprocessing
    approaches in the pipeline.
    """
    def handle_data(self,data: pd.DataFrame) -> Union[pd.DataFrame,pd.Series]:
        """
        Abstract method that must be implemented by concrete strategy classes
        
        This method should implement the specific data processing logic for
        the concrete strategy (e.g., preprocessing, splitting, etc.).
        
        Args:
            data (pd.DataFrame): Input data to be processed
        
        Returns:
            Union[pd.DataFrame, pd.Series]: Processed data in the appropriate format
        """
        pass

class DataPreprocessing(DataStrategy):
    """
    Concrete implementation for data preprocessing and cleaning
    
    This class handles all the data cleaning operations needed to prepare
    the raw house price data for model training. It includes handling missing
    values, removing duplicates, encoding categorical variables, and converting
    data types to the appropriate format.
    """

    def handle_data(self,dataset : pd.DataFrame) -> pd.DataFrame:
        """
        Comprehensive data preprocessing pipeline for house price data
        
        This method performs the following operations in sequence:
        1. Remove unnecessary columns that don't contribute to prediction
        2. Handle missing values in categorical and numerical columns
        3. Remove duplicate rows to ensure data quality
        4. Convert price strings to numerical values (Lac/Cr to rupees)
        5. Clean and convert area measurements to numerical format
        6. Encode categorical variables using LabelEncoder
        7. Remove any remaining rows with missing values
        
        Args:
            dataset (pd.DataFrame): Raw house price dataset from CSV file
        
        Returns:
            pd.DataFrame: Cleaned and preprocessed dataset ready for model training
        """
        # Step 1: Remove columns that don't contribute to prediction
        # These columns either contain redundant information or are not useful for modeling
        dataset.drop(['Index', 'Description', 'Ownership', 'overlooking','Car Parking','Dimensions','Plot Area','Super Area'], axis='columns', inplace=True)
        
        # Step 2: Handle missing values in categorical columns
        # Fill missing categorical values with 'Unknown' to preserve data
        column = ['Title','location','Transaction','Furnishing','facing']
        dataset[column] = dataset[column].fillna('Unknow')

        # Step 3: Handle missing values in numerical columns
        # Use mode (most frequent value) to fill missing bathroom and balcony counts
        mode_bathroom = dataset['Bathroom'].mode()
        mode_balcony = dataset['Balcony'].mode()

        dataset['Bathroom'] = dataset['Bathroom'].fillna(mode_bathroom)
        dataset['Balcony'] = dataset['Balcony'].fillna(mode_balcony)

        # Step 4: Remove duplicate rows to ensure data quality
        # Keep the first occurrence of any duplicate row
        duplicates_row = dataset[dataset.duplicated(keep='first')]
        duplicates_row.shape[0]  # Count of duplicates (for logging purposes)

        dataset.drop_duplicates(keep='first', inplace=True)

        # Step 5: Convert price strings to numerical values
        # Handle different price formats (Lac, Cr, plain numbers)
        dataset['Amount(in rupees)'] = dataset['Amount(in rupees)'].apply(self.convert_price)

        # Step 6: Clean and convert area measurements to numerical format
        # Remove units (sqft, sqm, sqyrd) and convert to numeric
        dataset['Carpet Area'] = dataset['Carpet Area'].astype(str)
        dataset['Carpet Area'] = dataset['Carpet Area'].str.replace('sqft', '', regex=True)
        dataset['Carpet Area'] = dataset['Carpet Area'].str.replace('sqm', '', regex=True)
        dataset['Carpet Area'] = dataset['Carpet Area'].str.replace('sqyrd', '', regex=True)
        dataset['Carpet Area'] = pd.to_numeric(dataset['Carpet Area'], errors='coerce')

        # Step 7: Ensure numerical columns are properly typed
        dataset['Bathroom'] = pd.to_numeric(dataset['Bathroom'], errors='coerce')
        dataset['Balcony'] = pd.to_numeric(dataset['Balcony'], errors='coerce')

        # Step 8: Encode categorical variables using LabelEncoder
        # This converts text categories to numerical values for model training
        label_encoder = LabelEncoder()
        test_to_numeric = ['Title','location','Transaction','Furnishing','facing','Status','Floor','Society']

        for x in test_to_numeric:
            dataset[x] = label_encoder.fit_transform(dataset[x])

        # Step 9: Remove any remaining rows with missing values
        # This ensures the final dataset is complete and ready for training
        dataset = dataset.dropna()

        return dataset
    
    def convert_price(self,amount):
        """
        Convert price strings to numerical values
        
        This method handles different price formats commonly found in Indian real estate:
        - 'Lac' represents lakhs (1 Lac = 100,000 rupees)
        - 'Cr' represents crores (1 Cr = 10,000,000 rupees)
        - Plain numbers are converted directly
        
        Args:
            amount (str): Price string in various formats (e.g., '2.5 Lac', '1.2 Cr', '500000')
        
        Returns:
            float: Price converted to rupees, or None if conversion fails
        """
        try:
            if 'Lac' in amount:
                # Convert lakhs to rupees (1 Lac = 100,000)
                amount = amount.replace('Lac', '').strip()
                return float(amount) * 100000
            elif 'Cr' in amount:
                # Convert crores to rupees (1 Cr = 10,000,000)
                amount = amount.replace('Cr', '').strip()
                return float(amount) * 10000000
            else:
                # Direct conversion for plain numbers
                return float(amount)
        except ValueError:
            # Return None if conversion fails (will be handled by dropna later)
            return None



class DataDivision(DataStrategy):
    """
    Concrete implementation for splitting data into training and testing sets
    
    This class handles the division of the preprocessed dataset into training
    and testing subsets. It also applies feature scaling to ensure all features
    are on the same scale for optimal model performance.
    """
    def handle_data(self,dataset : pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame,"X_train"],
    Annotated[pd.DataFrame,"X_test"],
    Annotated[pd.Series,"y_train"],
    Annotated[pd.Series,"y_test"],
]:
        """
        Split preprocessed data into training and testing sets with feature scaling
        
        This method performs the following operations:
        1. Select relevant features for model training
        2. Separate features (X) from target variable (y)
        3. Apply StandardScaler to normalize features
        4. Split data into training (67%) and testing (33%) sets
        5. Return the four datasets needed for model training and evaluation
        
        Args:
            dataset (pd.DataFrame): Preprocessed dataset ready for splitting
        
        Returns:
            Tuple containing:
            - X_train (pd.DataFrame): Scaled training features
            - X_test (pd.DataFrame): Scaled testing features
            - y_train (pd.Series): Training target variable (house prices)
            - y_test (pd.Series): Testing target variable (house prices)
        """
        # Step 1: Select features for model training
        # These features were identified as most relevant for house price prediction
        X = dataset[['Title', 'Bathroom', 'Carpet Area', 'location', 'Transaction', 'Furnishing', 'Balcony','facing','Price (in rupees)',
             'Status','Society','Floor']]
        
        # Step 2: Extract target variable (house prices)
        y = dataset['Amount(in rupees)']

        # Step 3: Apply StandardScaler to normalize features
        # This ensures all features are on the same scale (mean=0, std=1)
        # Scaling is crucial for algorithms like Random Forest to perform optimally
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Step 4: Convert scaled array back to DataFrame with original column names
        # This maintains the feature names for easier interpretation
        X = pd.DataFrame(X_scaled, columns=X.select_dtypes(include='number').columns)

        # Step 5: Split data into training (67%) and testing (33%) sets
        # The random_state ensures reproducible splits across different runs
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        return X_train, X_test, y_train, y_test