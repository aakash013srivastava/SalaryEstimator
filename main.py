from tkinter import ttk
from tkinter import Tk
from tkinter import messagebox
import os
import pandas as pd
import numpy as np
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


class SalaryEstimator():
    def __init__(self):
        self.root = Tk()
        self.root.title = "AutomatedSalaryEstimator"
        self.root.geometry("400x200")
        self.lbl = ttk.Label(self.root,text="Select File:")
        self.lbl.pack()
        self.dropdown = ttk.Combobox(self.root,values=self.getCSVFiles(),width='250')
        self.dropdown.pack()
        self.lbl_check_YOE = ttk.Label(self.root,text="Enter YOE for estimation:")
        self.lbl_check_YOE.pack()
        self.entry = ttk.Entry(self.root)
        self.entry.pack()
        self.btn = ttk.Button(self.root,text='Estimate',command=lambda: self.onSelectCombo(self.dropdown.get(),self.entry.get()))
        self.btn.pack()
        self.output_lbl = ttk.Label(self.root,text="")
        self.output_lbl.pack()

        if __name__ == '__main__':
            self.root.mainloop()

    def getCSVFiles(self):
        username = str(os.getlogin())
        path1 = "C:\\Users\\"+username+"\\Downloads\\*.csv"
        
        list_of_files = glob.glob(path1) 
        # print(list_of_files)
        return list_of_files
    
    def onSelectCombo(self,filename,yoe_to_predict):
        df = pd.read_csv(filename)
        print(df.columns)

        # Linear Regression ML code
        try:
            yoe = [x for x in df.columns if 'ears' in x ][0]
            salary = [x for x in df.columns if 'alary' in x][0]
            print(salary)
            df[yoe].fillna(0,inplace=True)
            X = df[[yoe]].values
            df[salary] = df[salary].replace('\$|,', '', regex=True)
            df[salary].fillna('',inplace=True)
            Y = df[[salary]].values


            imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
            imputer.fit(X)

            X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=1)
            regressor = LinearRegression()
            regressor.fit(X_train,Y_train)
            y_pred_test = regressor.predict([[float(yoe_to_predict)]])
            self.message = messagebox.showinfo('Salary:',y_pred_test)
            print(y_pred_test)
            # print('R2 score:'+r2_score(Y_test,y_pred))  
        except Exception as e:
            print(e)


    
SalaryEstimator()