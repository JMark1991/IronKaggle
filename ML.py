import pandas as pd 

sales= pd.read_csv("train.csv")

def month(s):
    m = s[5:7]
    return int(m)

def sales_weekday(df):
    df['month'] = df['Date'].apply(month)
    df.drop(columns=['Date','Unnamed: 0'], axis=1, inplace=True)
    dummy= pd.get_dummies(df)
    return dummy

def store_category(df):
    dfcut=pd.cut(df['Nb_customers_on_day'], 6, labels=['Incredibly Low',"Very Low","Low", "Medium","High",'Very High'])
    df['Sales Quantity']= dfcut
    df.drop(columns=["Store_ID"], axis=1, inplace=True)
    dummies= pd.get_dummies(df)
    return dummies


sales_dummies = sales_weekday(sales)
sales_dummies = store_category(sales_dummies)



X = sales_dummies[sales_dummies['Open']==1].drop('Sales', axis=1)
y = sales_dummies[sales_dummies['Open']==1]['Sales']


X_to_pred = sales_dummies.drop('Sales', axis=1)
y_to_pred = sales_dummies['Sales']


from sklearn.model_selection import train_test_split

#X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.05, random_state=42)


# Splitting the training data (Open = 1 removed)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

# Splitting the test data (all the rows)
X_to_pred_train, X_to_pred_test, y_to_pred_train, y_to_pred_test = train_test_split(X_to_pred, y_to_pred, train_size=0.85, random_state=42)



def Linear_reg_train(X_train, y_train):
    # Takes the training X and y
    # Prints the accuracy
    # Returns the model

    from sklearn.linear_model import LinearRegression

    # initialize model
    model = LinearRegression(n_jobs=-1, )

    # fit model
    model.fit(X_train, y_train)

    return model


def Ridge_reg_train(X_train, y_train):
    # Takes the training X and y
    # Prints the accuracy
    # Returns the model

    from sklearn.linear_model import Ridge

    # initialize model
    model = Ridge()

    # fit model
    model.fit(X_train, y_train)

    return model


def Random_forest_reg_train(X_train, y_train):
    # Takes the training X and y
    # Prints the accuracy
    # Returns the model

    from sklearn.ensemble import RandomForestRegressor

    # initialize model
    model = RandomForestRegressor(n_estimators=150, max_depth=16, n_jobs=-1)

    # fit model
    model.fit(X_train, y_train)

    return model


from sklearn.metrics import r2_score
model = Random_forest_reg_train(X_train, y_train)

#print(model.coef_)
#print(model.intercept_)

y_train_pred = model.predict(X_to_pred_train) * X_to_pred_train['Open']
print('Train accuracy: ', r2_score(y_to_pred_train, y_train_pred))

y_test_pred = model.predict(X_to_pred_test) * X_to_pred_test['Open']
print('Test accuracy: ', r2_score(y_to_pred_test, y_test_pred))

#######################################

sales = pd.read_csv('validation_features.csv')

def month(s):
    m = s[3:5]
    return int(m)

def sales_weekday(df):
    df['month'] = df['Date'].apply(month)
    df.drop(columns=['Date'], axis=1, inplace=True)
    dummy= pd.get_dummies(df)
    return dummy

sales_dummies = sales_weekday(sales)
sales_dummies = store_category(sales_dummies)

X = sales_dummies

y__pred = model.predict(X) * X['Open']

y__pred.to_csv('predictions.csv')