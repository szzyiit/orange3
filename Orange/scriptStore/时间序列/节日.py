# - 选择题参数名|中文描述|选项,格式为: [('选项1', 值, 是否默认), ('选项2', 值, 是否默认)]
# - 填空题参数名|中文描述|[默认值]
# 需要安装：pip install chinesecalendar
'''
根据节日构造相关特征
- cn|包括中国节日信息|[('是', True, True), ('否', False, False)]
- us|包括美国节日信息|[('是', True, False), ('否', False, True)]
- fldname|时间特征|['Month']
返回值: 
cn_holiday: 是否中国节日
days_after_cn_holiday: 中国节日后的几天
days_before_cn_holiday: 中国节日前的几天
cn_work_day: 是否工作日（考虑到了调休）
nth_succesive_day: 已经连续工作或者休息了多长时间
count: 这段是连续几天的工作或者休息日
'''

from typing import List
from pandas import DatetimeIndex
import numpy as np

import pandas as pd
from datetime import datetime
from Orange.data.pandas_compat import table_from_frame,table_to_frame


# 美国节日
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
# 中国节日
import chinese_calendar as cn_calendar

# the number of days from the previous holiday 
def days_after_holiday(date, holidays):
    difference=[]
    for holiday in holidays:
        if date < holiday:
            continue
        difference.append((date-holiday).days)
    if difference == []:
        return -1
    return difference[-1]

# the number of days to the next holiday
def days_before_holiday(date, holidays):
    for holiday in holidays:
        if date > holiday:
            continue
        return (holiday-date).days
    return  -1

def get_holidays(dates, include_cn_holidays=in_params['cn'], include_us_holidays=in_params['us'], fldname=in_params['fldname']):
    fld = dates[fldname]
    fld_dtype = fld.dtype
    if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        fld_dtype = np.datetime64
    
    # dates = pd.DataFrame({fldname:pd.date_range('2017-09-20', '2018-08-28')})

    if include_cn_holidays:
        cn_holidays: List = cn_calendar.get_holidays(start=dates[fldname].min(), end=dates[fldname].max(), include_weekends=False)
        cn_holidays: List = [ datetime.combine(d,  datetime.min.time()) for d in cn_holidays]

        dates['cn_holiday'] = dates[fldname].isin(cn_holidays)



        dates['days_after_cn_holiday']= dates.apply(lambda row: days_after_holiday(row[fldname], cn_holidays), axis=1)
        dates['days_before_cn_holiday']= dates.apply(lambda row: days_before_holiday(row[fldname], cn_holidays), axis=1)
        dates['date_datetime'] = dates[fldname].dt.to_pydatetime()
        dates['cn_work_day']= dates.apply(lambda row: cn_calendar.is_workday(row[fldname]), axis=1)
        dates.drop('date_datetime', axis=1, inplace=True)

        '''
        This uses the compare-cumsum-groupby pattern to find the contiguous groups, 
        because df[2].diff().ne(0) gives us a True whenever a value isn't the same as the previous, 
        and the cumulative sum of those gives us a new number whenever a new group of 1s starts.

        This will mean that we have the same group_id for binary values crossing different names, 
        of course, but since we're grouping on both df[0] (the names) and group_ids, we're okay.
        ref: https://stackoverflow.com/questions/50430148/pandas-cumulative-sum-of-consecutive-ones
        '''
        group_ids = dates.cn_work_day.diff().ne(0).cumsum()
        dates['cn_work_day_int'] = dates['cn_work_day'].astype(int) - 0.5
        dates["nth_succesive_day"] = abs(dates.cn_work_day_int.groupby([group_ids]).cumsum() * 2)
        dates.drop('cn_work_day_int', axis=1, inplace=True)
        dates['count'] = dates.groupby(group_ids)['cn_work_day'].transform('count')

    if include_us_holidays:
        cal = calendar()
        us_holidays: DatetimeIndex = cal.holidays(start=dates[fldname].min(), end=dates[fldname].max())

        # a column titles “holiday” in our data frame 
        # which contains True if the date is a US federal holiday 
        # and False if it is not.
        dates['us_holiday'] = dates[fldname].isin(us_holidays)
        dates['days_after_us_holiday']= dates.apply(lambda row: days_after_holiday(row[fldname], us_holidays), axis=1)
        dates['days_before_us_holiday']= dates.apply(lambda row: days_before_holiday(row[fldname], us_holidays), axis=1)
    return dates

dates = table_to_frame(in_data, include_metas=True)

out_dates = get_holidays(dates)

out_data = table_from_frame(out_dates)

print(out_dates.head())

