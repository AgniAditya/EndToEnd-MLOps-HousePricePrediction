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
    Abstract class defining startegy for handling data
    """

    def handle_data(self,data: pd.DataFrame) -> Union[pd.DataFrame,pd.Series]:
        pass

class DataPreprocessing(DataStrategy):
    """
    Pre process the Data for model training
    """

    def handle_data(self,dataset : pd.DataFrame) -> pd.DataFrame:
        """
        These are the steps i have take to clean the dataset and prepare it for our training model
        to get the maximum accuracy

        Args:
            dataset : Data which need to be clean and prepare for the model
        
        Retruns:
            DataFrame : It retruns a Dataframe of the given dataset after applying the below operations.
        """

        dataset.drop(['Index', 'Description', 'Ownership', 'overlooking','Car Parking','Dimensions','Plot Area','Super Area'], axis='columns', inplace=True)
        
        column = ['Title','location','Transaction','Furnishing','facing']
        dataset[column] = dataset[column].fillna('Unknow')

        mode_bathroom = dataset['Bathroom'].mode()
        mode_balcony = dataset['Balcony'].mode()

        dataset['Bathroom'] = dataset['Bathroom'].fillna(mode_bathroom)
        dataset['Balcony'] = dataset['Balcony'].fillna(mode_balcony)

        duplicates_row = dataset[dataset.duplicated(keep='first')]
        duplicates_row.shape[0]

        dataset.drop_duplicates(keep='first', inplace=True)

        dataset['Amount(in rupees)'] = dataset['Amount(in rupees)'].apply(self.convert_price)

        dataset['Carpet Area'] = dataset['Carpet Area'].astype(str)
        dataset['Carpet Area'] = dataset['Carpet Area'].str.replace('sqft', '', regex=True)
        dataset['Carpet Area'] = dataset['Carpet Area'].str.replace('sqm', '', regex=True)
        dataset['Carpet Area'] = dataset['Carpet Area'].str.replace('sqyrd', '', regex=True)
        dataset['Carpet Area'] = pd.to_numeric(dataset['Carpet Area'], errors='coerce')

        dataset['Bathroom'] = pd.to_numeric(dataset['Bathroom'], errors='coerce')
        dataset['Balcony'] = pd.to_numeric(dataset['Balcony'], errors='coerce')

        label_encoder = LabelEncoder()
        test_to_numeric = ['Title','location','Transaction','Furnishing','facing','Status','Floor','Society']

        for x in test_to_numeric:
            dataset[x] = label_encoder.fit_transform(dataset[x])

        dataset = dataset.dropna()

        return dataset
    
    def convert_price(self,amount):
        try:
            if 'Lac' in amount:
                amount = amount.replace('Lac', '').strip()
                return float(amount)
            elif 'Cr' in amount:
                amount = amount.replace('Cr', '').strip()
                return float(amount) * 100
            else:
                return float(amount)
        except ValueError:
            return None



class DataDivision(DataStrategy):
    """
    Divides data into Training and Testing part
    """
    def handle_data(self,dataset : pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame,"X_train"],
    Annotated[pd.DataFrame,"X_test"],
    Annotated[pd.Series,"y_train"],
    Annotated[pd.Series,"y_test"],
]:
        """
        Args:
            dataset : Dataframe of the Data which will divide into training and testing parts

        Returns:
            It returns four values X_train, X_test, y_train, y_test. Training data are X_train,Y_train.
            Testing data are x_test,y_test.
        """

        X = dataset[['Title', 'Bathroom', 'Carpet Area', 'location', 'Transaction', 'Furnishing', 'Balcony','facing','Price (in rupees)',
             'Status','Society','Floor']]
        y = dataset['Amount(in rupees)']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X = pd.DataFrame(X_scaled, columns=X.select_dtypes(include='number').columns)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        return X_train, X_test, y_train, y_test