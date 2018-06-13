p = '/Users/numlaut/dsi/projects/7.0 project 4/West_Nile_Kaggle/assets/input/'
spraydf=pd.read_csv(p+'spray.csv')
traindf=pd.read_csv(p+'train.csv')
testdf = pd.read_csv(p+'test.csv')
weatherdf=pd.read_csv(p+'clean_weather.csv')

weather = weatherdf.drop('Unnamed: 0',axis=1)
# validation for next step:
print(weather.shape)
mask = (weather['Station'] == 1)
weather = weather[mask]
print(weather.shape)
#weather = weather.drop('Station', axis=1)
weather['utc']=weather['Date'].map(date_to_utc)
train = traindf
train['utc']=traindf['Date'].map(date_to_utc)

test2 = testdf.rename(columns=renamed)
test2['utc']=test2['date'].map(date_to_utc)

renamed = {'Date':'date'}
weather = weather.rename(columns=renamed)
train = train.rename(columns=renamed)


weather['utc'] = weather['utc'].map(lambda x: x-(8*24*60*60))
weather['date']= weather['utc'].map(utc_to_date)

import time
def date_to_utc(x):
    return int(time.mktime(time.strptime(x,'%Y-%m-%d')))

def utc_to_date(x):
    return time.strftime('%Y-%m-%d',time.gmtime(x))


test2 = test2.merge(weather,on=['utc','date'],how='left',copy=True,validate='m:1')
test2 = test2.drop('SnowFall',axis=1)
test2 = test2.dropna(axis=0)

new = train.merge(weather,on=['utc','date'],how='left',copy=True,validate='m:1')
new = new.drop('SnowFall',axis=1)
new = new.dropna(axis=0)
new.shape

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
X = new.drop(['date','utc','WnvPresent','NumMosquitos','AddressNumberAndStreet','Latitude','Longitude','CodeSum','Sunrise'],axis=1)
y = new['WnvPresent']

Xtest2 = test2.drop(['date','utc','AddressNumberAndStreet','Latitude','Longitude','CodeSum','Sunrise'],axis=1)
#ytest2 = test2['WnvPresent']


Xfeats = list(new.columns)

Xtrain,Xtest,ytrain,ytest = train_test_split(X,y,stratify=y)

ss = StandardScaler()
Xtrain = ss.fit_transform(Xtrain)
Xtest = ss.transform(Xtest)

##################################################################################################################################

def scores(model):
    return model.score(Xtrain,ytrain),model.score(Xtest,ytest),model.score(Xtest2)

from sklearn.dummy import DummyClassifier
base_strat = DummyClassifier().fit(Xtrain,ytrain)
print(scores(base_strat))

base_majority = DummyClassifier(strategy='constant',constant=0).fit(Xtrain,ytrain)
print(scores(base_majority))
