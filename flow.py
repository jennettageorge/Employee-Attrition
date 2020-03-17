
from metaflow import FlowSpec, step, catch, retry, IncludeFile, Parameter
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix
import itertools
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from metaflow import FlowSpec, step, catch, retry, IncludeFile, Parameter
import pandas as pd
from sklearn.metrics import f1_score

class EmployeeAttritioonPredictor(FlowSpec):


    @step
    def start(self):


        self.employees = pd.read_csv('data/Employees.csv')
        self.surveys = pd.read_csv('data/Survey.csv')
        self.performance = pd.read_csv('data/Performance.csv')

        self.next(self.clean_data)




    @step
    def clean_data(self):
        #Employees Table
        import datetime

        '''function to turn a datetime pd.Series into a numerical feature'''
        def date_to_feat(series):


            dtseries = series.apply(lambda x : datetime.datetime.strptime(x, '%m/%d/%y'))
            epoch = datetime.datetime.utcfromtimestamp(0)
            featseries = dtseries.apply(lambda x : (x - epoch).total_seconds()/86400)
            return featseries

        #Deal with null values
        self.employees.EmploymentEndReason = self.employees.EmploymentEndReason.apply(lambda x: 0 if str(x) == 'nan' else 1)
        self.employees.NumPreviousCompanies = self.employees.NumPreviousCompanies.fillna(0)

        #Deal with Dates
        self.employees['EmploymentStartTime'] = date_to_feat(self.employees.EmploymentStartDate)
        self.employees.drop('EmploymentStartDate', axis=1,inplace=True)
        self.employees.drop('EmploymentEndDate',axis=1,inplace=True)

        self.performance['ReviewTime'] = date_to_feat(self.performance.ReviewDate)
        self.performance.drop('ReviewDate', axis=1, inplace=True)


        #Deal with categorical fields
        is_categorical = self.employees[['DegreeCompleted','DegreeField','Department','Gender','MaritalStatus','TravelFrequency']]

        cat_dummies = pd.get_dummies(is_categorical)

        self.employees = self.employees.join(cat_dummies)
        for col in list(is_categorical.columns):
            self.employees = self.employees.drop(col, axis=1)

        #Survey Table
        # Cleaning the survey data before we join tables
        self.surveys.Response = self.surveys.Response.apply(lambda x: 0 if (x == 'Very Unsatisfied' or x =='Very Poor')\
                                      else (1 if (x == 'Somewhat Unsatisfied' or x == 'Poor')\
                                      else (2 if (x == 'Neither Satisfied nor Unsatisfied' or x == 'Fair')\
                                      else (3 if (x == 'Somewhat Satisfied' or x == 'Good')\
                                      else (4 if (x == 'Very Satisfied' or x =='Excellent') else 5)))))
        Q1 = self.surveys[self.surveys.QuestionNum == 'Q1']
        Q2 = self.surveys[self.surveys.QuestionNum == 'Q2']
        Q3 = self.surveys[self.surveys.QuestionNum == 'Q3']
        Q4 = self.surveys[self.surveys.QuestionNum == 'Q4']

        Q1.rename(columns={"Response":"ResponseQ1"}, inplace = True)
        Q2.rename(columns={"Response":"ResponseQ2"}, inplace = True)
        Q3.rename(columns={"Response":"ResponseQ3"}, inplace = True)
        Q4.rename(columns={"Response":"ResponseQ4"}, inplace = True)

        Q1 = Q1.set_index(self.employees.EmployeeId).drop('EmployeeId',axis=1)
        Q2 = Q2.set_index(self.employees.EmployeeId).drop('EmployeeId',axis=1)
        Q3 = Q3.set_index(self.employees.EmployeeId).drop('EmployeeId',axis=1)
        Q4 = Q4.set_index(self.employees.EmployeeId).drop('EmployeeId',axis=1)

        largetable = pd.concat([Q1,Q2,Q3,Q4], sort=False)
        self.responses = largetable.groupby('EmployeeId').sum()

        self.employees = self.employees.set_index('EmployeeId')
        self.emp = self.employees.join(self.responses)
        self.emp = self.emp.join(self.performance.iloc[:,1:])
        self.labels = self.emp.EmploymentEndReason
        self.emp.drop('EmploymentEndReason', axis=1,inplace=True)

        self.next(self.calculate_vif_)

    @step
    def calculate_vif_(self ):
        thresh=5.0
        variables = list(range(self.emp.shape[1]))
        dropped = True
        while dropped:
            dropped = False
            vif = [variance_inflation_factor(self.emp.iloc[:, variables].values, ix)
                   for ix in range(self.emp.iloc[:, variables].shape[1])]

            maxloc = vif.index(max(vif))
            if max(vif) > thresh:
                print('dropping \'' + self.emp.iloc[:, variables].columns[maxloc] +
                      '\' at index: ' + str(maxloc))
                del variables[maxloc]
                dropped = True

        print('Remaining variables:')
        print(self.emp.columns[variables])

        self.next(self.calculate_corr_)

    @step
    def calculate_corr_(self):
        corr = self.emp.corr()
        colsbefore = set(self.emp.columns)
        columns = np.full((corr.shape[0],), True, dtype=bool)
        for i in range(corr.shape[0]):
            for j in range(i+1, corr.shape[0]):
                if corr.iloc[i,j] >= 0.8:
                    if columns[j]:
                        columns[j] = False
        selected_columns = self.emp.columns[columns]
        self.emp = self.emp[selected_columns]
        colsafter = set(self.emp.columns)

        print('# of Columns after removal: ', len(list(self.emp.columns)))
        print('Columns Removed: ', colsbefore - colsafter)


        self.next(self.predict)
    @step
    def predict(self):
        #train test split
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.emp, self.labels, test_size=0.05, stratify=self.labels, random_state=0)

        # Train A Decision Tree Model
        # Create decision tree classifer object
        clf = RandomForestClassifier(random_state=0, n_jobs=-1)

        # Train model
        model = clf.fit(self.x_train, self.y_train)

        #Predict on testing data
        self.rf_predictions = clf.predict(self.x_test)

        self.next(self.metrics)

    @step
    def metrics(self):
        # Evaluate predictions
        accuracy = accuracy_score(self.y_test, self.rf_predictions)
        print("Accuracy: %.2f%%" % (accuracy * 100.0))
        f1 = f1_score(self.y_test, self.rf_predictions)
        print("F1 Score: .2f%%" % (f1 ))
        cm = confusion_matrix(self.y_test, self.rf_predictions)
        print('Confusion Matrix: \n' , cm)

        self.next(self.end)
    '''
    @step
    def predict(self):
        #train test split
        x_train, x_test, y_train, y_test = train_test_split(self.emp, self.labels, test_size=0.2, stratify=self.labels, random_state=0)

        # Create the parameter grid based on the results of random search
        param_grid = {
            'bootstrap': [True],
            'max_depth': [30,40,50, 60],
            'max_features': [2, 3, 4],
            'min_samples_leaf': [4, 5],
            'min_samples_split': [8, 10, 12],
            'n_estimators': [50,100, 200, 300]
        }
        # Create a based model
        rf = RandomForestClassifier()
        # Instantiate the grid search model
        grid_search = GridSearchCV(estimator = rf, param_grid = param_grid,
                                  cv = 3, n_jobs = -1, verbose = 2)
        # Fit the grid search to the data
        grid_search.fit(x_train, y_train)
        grid_search.best_params_
        best_grid = grid_search.best_estimator_
        print(best_grid)
        predictions = best_grid.predict(x_test)

        # evaluate predictions
        accuracy = accuracy_score(y_test, predictions)
        print("Accuracy: %.2f%%" % (accuracy * 100.0))

        self.next(self.end)
        '''
    @step
    def end(self):
        print('Ending')

if __name__ == '__main__':
        EmployeeAttritioonPredictor()
