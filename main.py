from tkinter import ttk
from tkinter import Tk
from tkinter import messagebox
import os
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,NavigationToolbar2Tk)
from scipy.stats import boxcox
from scipy.stats.mstats import normaltest
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import traceback

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

        self.lbl_check_age = ttk.Label(self.root,text="Enter Age for estimation:")
        self.lbl_check_age.pack()
        self.entry_age = ttk.Entry(self.root)
        self.entry_age.pack()

        self.lbl_gender = ttk.Label(self.root,text="Select Gender:")
        self.lbl_gender.pack()
        self.dropdown_gender = ttk.Combobox(self.root,values=['Male','Female'],width='100')
        self.dropdown_gender.pack()

        self.btn = ttk.Button(self.root,text='Estimate',command=lambda: self.onSelectCombo(self.dropdown.get(),self.entry.get(),self.entry_age.get(),self.dropdown_gender.get()))
        self.btn.pack()
        self.output_lbl = ttk.Label(self.root,text="")
        self.output_lbl.pack()

        if __name__ == '__main__':
            self.root.mainloop()

    def getCSVFiles(self):
        username = str(os.getlogin())
        path1 = "C:\\Users\\"+username+"\\Downloads\\*.csv"
        list_of_files = glob.glob(path1) 
        return list_of_files
    
    
    
    def onSelectCombo(self,filename,yoe_to_predict,age_to_predict,gender):
        df = pd.read_csv(filename)
        # k2,p = normaltest(df[['Salary']])
        # print(p)
        # if p < 0.05:
        #     b = boxcox(df[['Salary']].values.ravel())
        #     print((b))
        #     # k2_2,p_2 = normaltest(b)
        #     # print(p_2)
        #     messagebox.showinfo("Message after transform:",p_2)
        # else:
        #     messagebox.showinfo("boxcox not required:",p)

        
        # Regression ML code
        try:
            yoe = [x for x in df.columns if 'years' in x.lower() ][0]
            salary = [x for x in df.columns if 'salary' in x.lower()][0]
            age = [x for x in df.columns if 'age' in x.lower()][0]

            gen = [x for x in df.columns if 'gender' in x.lower()][0]
            gen_cols_cat = pd.get_dummies(df[gen])
            df.drop(gen,axis=1,inplace=True)
            df2 = pd.concat([df,gen_cols_cat],axis=1)
            print(df2.head())
            # df2 = pd.concat([df[[yoe,gen]],gen_cols_cat],axis=1)
            # messagebox.showinfo('test:',df2.head())
            df2[salary] = df[salary].replace('\$|,', '', regex=True)
            df2[salary].fillna('',inplace=True)

  
            gender_determiner = 0 if gender == 'Female' else 1
            X = df2[[yoe,age,'Female']]
            Y = df2[[salary]]
            

            # Handle Missing values
            imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
            imputer.fit(X)


            X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=12)
            regressor = LinearRegression()
            regressor.fit(X_train,Y_train)
            y_pred_test = regressor.predict([[float(yoe_to_predict),float(age_to_predict),gender_determiner]])
            messagebox.showinfo('Salary:',y_pred_test)
            print(y_pred_test)
            # print('R2 score:'+r2_score(Y_test,y_pred))  
        except Exception as e:
            traceback.print_exc()


    
SalaryEstimator()