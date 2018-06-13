from sklearn.base import TransformerMixin

class SprayFactor(TransformerMixin):
    """
    building a feature to represent the effect of anti-mosquito
    spray on collections based on time and location distances
    """

    def __init__(self, spraydf, days_decay=20, dist_decay=8,
                timename='utc',locname='loc', dist_unit='mi',
                dist_func=euclidean):
        """
        decay_days set to 20 due to permethrin spray research that suggests 15-20 day effective range (aqua reslin ULV)

        note: decay_dist currently set to represent half of mean distance (16mi) between collection points (radius basically)
              we can change/tune this at some point if we want to
        """

        self.spray = spraydf
        self.days_decay = days_decay
        self.dist_decay = dist_decay
        self.dist_unit = dist_unit
        self.dist_func = dist_func
        return None

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        loc_coef_column = self.loc_coef(X)
        time_coef_column = self.time_coef(X)
        Xout = X
        Xout['loc_coef'] = loc_coef_column
        Xout['time_coef'] = time_coef_column
        return Xout

    def filter_by_utc(self, date):
        """1: throw out any spray data that's from the future.  to help w/calculation efficiency
           2: throw out any spray data that's excessively in the past."""
        mask = (self.spray['utc'] <= date)
        outdf = self.spray[mask]
        mask2 = (diff(date,outdf['utc'].values) <= self.days_decay)
        return outdf[mask2]

    def time_coef(self, X):
        time_coef_matrix = []
        for row in X.values:
            time_coefs = []
            time_i = X.columns.get_loc('utc')
            date = row[time_i]
            date = np.int(date)
            spray = self.filter_by_utc(date)
            #print('date: ', row[0], 'spray locs: ',spray.shape[0])

            for tloc in spray['utc'].values:
                x = self.timediff(date,tloc) #diff function

                if x <= self.days_decay:
                    time_coefs.append(self.decay(x, self.days_decay))
                #else:
                #    time_coefs.append(0)
            coef = np.mean(time_coefs) #take mean
            if coef != coef: coef = 0 # impute 0 for nulls

            time_coef_matrix.append(coef) #add to matrix
        #end for loop
        return np.array(time_coef_matrix)

    def loc_coef(self, X):
        """
        for each row in df, calculate aggregate coef representing
        distance from recent,non-future spray locations
        """

        loc_coef_matrix = [] #output

        for row in X.values: # for each time in X
            loc_coefs = []
            utc_i = X.columns.get_loc('utc') #index of utc for array
            loc_i = X.columns.get_loc('loc') #index of loc for array
            date = np.int(row[utc_i])
            spray = self.filter_by_utc(date) #sprays within 20 days

            for sloc in spray['loc'].values:
                x = self.dist_func(row[loc_i], sloc) #diff function

                if x <= self.dist_decay: #throw out if outside "limit"
                    loc_coefs.append(self.decay(x, self.dist_decay))
                else:
                    loc_coefs.append(0)

            coef = np.mean(loc_coefs) # other stat function could go here
            if coef != coef: coef = 0 # impute 0 for nulls

            loc_coef_matrix.append(coef) #add to matrix
        # / end for loop
        return np.array(loc_coef_matrix)

    def decay(self, x, end):
        """ sigmoid decay function y~=1 at x==0, and y~=0 at x==end """
        decay_end = end #our point where we want y to decay to 0
        decay_width = end #width of the curve.  currently set wide, can tighten for more 'step' like func

        midpoint = decay_end - decay_width/2
        scale = 10/decay_width

        transformed = -1/(1+np.exp(-scale*(x-midpoint)))+1

        return transformed
