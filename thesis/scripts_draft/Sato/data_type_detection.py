import pandas as pd
from uuid import UUID
import datetime

def str_or_numeric(val):
  try:
    without_null = val.map(eval)
    return without_null.dtypes.name
  except:
    return False

def is_valid_uuid(val):
    try:
        UUID(val)
        return True
    except:
        return False

def is_date(val):
  fmt = ['%Y-%m-%d', '%Y/%m/%d', '%d.%m.%Y', '%d/%m/%Y', '%Y-%m-%dT%H:%M:%SZ', '%B %d, %Y', '%Y-%m-%d %H:%M:%S']
  for f in fmt:
    try:
      datetime.datetime.strptime(val, f)
      return True
    except:
      pass
  return False

def data_type(df):
    df_dtypes = pd.DataFrame(df.dtypes)
    for i in range(len(df_dtypes)):
        df_dtypes[0][i] = df_dtypes[0][i].name
    df_dtypes = df_dtypes.T
    for c in df_dtypes.columns:
        val = df[df[c] != ''][c]
        # eval function -> check numeric values
        if df_dtypes[c][0] == 'object':
            new_type = str_or_numeric(val)
            if len(val) != 0 and new_type is not False:
                df_dtypes[c][0] = new_type
            # is_valid_uuid -> check uuid type
            else:
                valid_uuid = val.map(is_valid_uuid)
                if len(valid_uuid) != 0 and valid_uuid.eq(True).all():
                    df_dtypes[c][0] = 'uuid'
                # is_date -> check date type
                else:
                    valid_date = val.map(is_date)
                    if len(valid_date) != 0 and valid_date.eq(True).all():
                        df_dtypes[c][0] = 'date'
                    else:
                        df_dtypes[c][0] = 'string'
    return df_dtypes.iloc[0].values.tolist()