"""
量化选股策略（该部分docstring如非必要请做保留，有利于未来的策略调整和debug）

本策略基于财务健康度和市场流动性筛选股票，旨在选取优质证券构建投资组合。策略流程如下：

1. **流动性筛选**：
   - 剔除最近半年日均成交金额排名后20%的证券，以确保选取的股票具备足够的市场流动性。

2. **行业剔除**：
   - 剔除国证行业分类标准下属于金融或房地产行业的证券，以减少行业系统性风险对策略的影响。

3. **财务稳定性筛选**：
   - 剔除近12个季度ROE（净资产收益率）稳定性排名后10%的证券，以确保所选公司具备较强的盈利能力和持续性。

4. **财务健康度筛选**：
   - 选取近一年自由现金流（FCF）、企业价值（EV）和近三年经营活动现金流均为正的证券，以确保公司具备良好的现金流状况。
   - 剔除近一年经营活动现金流占营业利润比例排名后30%的证券，以筛选出现金流质量较高的公司。

5. **排序与选股**：
   - 对剩余证券按照近一年自由现金流率（FCF/EV）从高到低排序。
   - 选取排名前100的证券作为最终指数样本。

该策略通过流动性、行业筛选、财务稳定性及现金流筛选等多个维度，力求构建一个稳定且优质的投资组合，以提高长期回报并降低风险。

PS：该代码默认在Google Colab上运行
"""

!pip install tushare
!pip install tqdm

import os
import pickle
import tushare as ts
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
from google.colab import userdata
from google.colab import drive

# 定义缓存目录，并确保目录存在
cache_dir = './cache'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

# 辅助函数：检查 DataFrame 是否包含必要字段
def check_required_columns(df, required_cols, df_name="DataFrame"):
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"{df_name} 缺少必要的字段: {missing_cols}. 请检查API返回字段或调整调用的字段名称。")

# 获取 Tushare API Token
tushare_api = userdata.get('TS_5')
ts.set_token(tushare_api)
pro = ts.pro_api()

# 新增统一时间区间参数：用于财务数据、基本面数据及回测时间区间
TIME_RANGE_START = '20150101'
TIME_RANGE_END = '20241231'

# === Step 1: 获取数据 ===

# 1.1 获取全部股票列表（获取全市场股票）
stock_list = pro.stock_basic(exchange='', list_status='L', fields='ts_code, symbol, name, industry')
codes = stock_list['ts_code'].tolist()

# 1.2 获取日均成交数据，并缓存
trade_data_file = os.path.join(cache_dir, "trade_data.pkl")
if os.path.exists(trade_data_file):
    with open(trade_data_file, 'rb') as f:
        trade_data = pickle.load(f)
    print("加载缓存的日均成交数据")
    
    # 新增逻辑：检查缓存中最后的交易日期
    if not trade_data.empty and 'trade_date' in trade_data.columns:
        last_trade_date = trade_data['trade_date'].max()
        print(f"缓存中最后交易日期：{last_trade_date}")
    else:
        last_trade_date = None
    
    # 获取交易日历，并筛选出比缓存中最后日期更新的日期
    cal_df = pro.trade_cal(exchange='SSE', is_open='1',
                           start_date=last_trade_date if last_trade_date else TIME_RANGE_START,
                           end_date=TIME_RANGE_END, fields='cal_date')
    all_trade_dates = cal_df['cal_date'].tolist()
    if last_trade_date:
        new_trade_dates = [d for d in all_trade_dates if d > last_trade_date]
    else:
        new_trade_dates = all_trade_dates
    
    # 增量下载新数据（如果有更新）
    if new_trade_dates:
        def get_daily_basic(trade_date, retries=3):
            for _ in range(retries):
                try:
                    df = pro.daily_basic(trade_date=trade_date,
                                         fields='ts_code, trade_date, turnover_rate_f, total_mv, float_mv, circ_mv, pe, pb, dv_ratio, total_share')
                    return df
                except Exception as e:
                    print(f"获取 {trade_date} 数据时出错: {e}")
                    time.sleep(1)
            return pd.DataFrame()
    
        trade_data_list = []
        for date in tqdm(new_trade_dates, desc="增量获取日均成交金额数据"):
            df = get_daily_basic(date)
            if not df.empty:
                check_required_columns(df, ['turnover_rate_f', 'total_mv'], df_name=f"日均成交数据-{date}")
                trade_data_list.append(df)
            time.sleep(1)
        if trade_data_list:
            new_trade_data = pd.concat(trade_data_list, ignore_index=True)
            trade_data = pd.concat([trade_data, new_trade_data], ignore_index=True)
            with open(trade_data_file, 'wb') as f:
                pickle.dump(trade_data, f)
            print("更新并保存日均成交数据到缓存")
        else:
            print("没有新增日均成交数据")
    else:
        print("缓存已是最新，无需更新")
    
else:
    # 利用交易日历获取所有有效的交易日
    cal_df = pro.trade_cal(exchange='SSE', is_open='1', start_date=TIME_RANGE_START, end_date=TIME_RANGE_END, fields='cal_date')
    trade_dates = cal_df['cal_date'].tolist()
    
    # 定义带重试机制的辅助函数
    def get_daily_basic(trade_date, retries=3):
        for _ in range(retries):
            try:
                df = pro.daily_basic(trade_date=trade_date,
                                     fields='ts_code, trade_date, turnover_rate_f, total_mv, float_mv, circ_mv, pe, pb, dv_ratio, total_share')
                return df
            except Exception as e:
                print(f"获取 {trade_date} 数据时出错: {e}")
                time.sleep(1)
        return pd.DataFrame()
    
    trade_data_list = []
    for date in tqdm(trade_dates, desc="获取日均成交金额数据"):
        df = get_daily_basic(date)
        if not df.empty:
            check_required_columns(df, ['turnover_rate_f', 'total_mv'], df_name=f"日均成交数据-{date}")
            trade_data_list.append(df)
        time.sleep(1)
    trade_data = pd.concat(trade_data_list, ignore_index=True) if trade_data_list else pd.DataFrame()
    with open(trade_data_file, 'wb') as f:
        pickle.dump(trade_data, f)
    print("保存日均成交数据到缓存")

# 计算成交金额，并求均值
trade_data = trade_data[trade_data['turnover_rate_f'] != 0]
trade_data['amt'] = trade_data['total_mv'] / trade_data['turnover_rate_f']
daily_avg_amt = trade_data.groupby("ts_code")["amt"].mean().reset_index()

# 1.3 行业信息已在 stock_list 中

# 1.4 获取 ROE 数据，并缓存
roe_data_file = os.path.join(cache_dir, "roe_data.pkl")
if os.path.exists(roe_data_file):
    with open(roe_data_file, 'rb') as f:
        roe_data = pickle.load(f)
    print("加载缓存的ROE数据")
else:
    # 使用 VIP 接口批量获取 ROE 数据
    roe_data = pro.fina_indicator_vip(start_date=TIME_RANGE_START, end_date=TIME_RANGE_END,
                                      fields='ts_code, ann_date, roe')
    # 获得每只股票最近12个季度数据
    roe_data = roe_data.sort_values(by='ann_date', ascending=False)
    roe_data = roe_data.groupby('ts_code').head(12)
    # 计算每只股票的 ROE 标准差
    roe_data['roe_std'] = roe_data.groupby("ts_code")["roe"].transform(np.std)
    # 预处理：每只股票保留最新记录
    roe_data = roe_data.sort_values('ann_date', ascending=False).groupby('ts_code').head(1)
    with open(roe_data_file, 'wb') as f:
        pickle.dump(roe_data, f)
    print("保存ROE数据到缓存")

# 1.5 获取现金流、资产负债、利润数据，并缓存（使用 VIP 接口一次性获取全市场数据）
cashflow_file = os.path.join(cache_dir, "cashflow_data.pkl")
balance_file  = os.path.join(cache_dir, "balance_data.pkl")
income_file   = os.path.join(cache_dir, "income_data.pkl")

if os.path.exists(cashflow_file) and os.path.exists(balance_file) and os.path.exists(income_file):
    with open(cashflow_file, 'rb') as f:
        cashflow_data = pickle.load(f)
    with open(balance_file, 'rb') as f:
        balance_data = pickle.load(f)
    with open(income_file, 'rb') as f:
        income_data = pickle.load(f)
    print("加载缓存的现金流、资产负债和利润数据")
else:
    # 使用 VIP 接口一次性获取全市场数据
    cashflow_data = pro.cashflow_vip(start_date=TIME_RANGE_START, end_date=TIME_RANGE_END,
                                     fields='ts_code, ann_date, free_cashflow, n_cashflow_act')
    balance_data  = pro.balancesheet_vip(start_date=TIME_RANGE_START, end_date=TIME_RANGE_END,
                                         fields='ts_code, ann_date, total_assets, total_liab')
    income_data   = pro.income_vip(start_date=TIME_RANGE_START, end_date=TIME_RANGE_END,
                                    fields='ts_code, ann_date, operate_profit')
    with open(cashflow_file, 'wb') as f:
        pickle.dump(cashflow_data, f)
    with open(balance_file, 'wb') as f:
        pickle.dump(balance_data, f)
    with open(income_file, 'wb') as f:
        pickle.dump(income_data, f)
    print("保存现金流、资产负债和利润数据到缓存")

# 预处理资产负债数据：取最新一期记录
if not balance_data.empty:
    balance_data['ev'] = balance_data['total_assets'] - balance_data['total_liab']
    balance_data_latest = balance_data.sort_values('ann_date').groupby('ts_code').tail(1)
else:
    balance_data_latest = pd.DataFrame()

# 预处理现金流数据：每只股票保留最新记录
if not cashflow_data.empty:
    cashflow_data = cashflow_data.sort_values('ann_date').groupby('ts_code').tail(1)

# 预处理利润数据：每只股票保留最新记录
if not income_data.empty:
    income_data = income_data.sort_values('ann_date').groupby('ts_code').tail(1)

# 将现金流数据与资产负债数据合并，计算自由现金流率（free_cashflow / ev）
if not cashflow_data.empty and not balance_data_latest.empty:
    cashflow_data = cashflow_data.merge(balance_data_latest[['ts_code', 'ev']], on="ts_code", how="left")
    cashflow_data['fcf_rate'] = cashflow_data['free_cashflow'] / cashflow_data['ev']

# 合并利润数据（使用最新的营业利润数据），计算经营现金流占营业利润比例
if not income_data.empty:
    print("income_data 的列：", income_data.columns)
    if 'operate_profit' in income_data.columns:
        cashflow_data = cashflow_data.merge(income_data[['ts_code', 'operate_profit']], on="ts_code", how="left")
        check_required_columns(cashflow_data, ['n_cashflow_act'], df_name="合并后现金流数据")
        check_required_columns(cashflow_data, ['operate_profit'], df_name="合并后利润数据")
        cashflow_data['cash_to_op_income'] = cashflow_data['n_cashflow_act'] / cashflow_data['operate_profit']
    else:
        print("警告：利润数据中缺少 'operate_profit' 字段，将无法计算经营现金流与营业利润的比例。")
        cashflow_data['cash_to_op_income'] = np.nan
else:
    cashflow_data['cash_to_op_income'] = np.nan

# ---------------- 新增函数 ----------------
def select_stocks(cutoff_date):
    """
    根据截止日期（避免前视）执行选股逻辑
    （demo中未做严格的截止日过滤，仅封装原有逻辑）
    """
    # 2.1 流动性筛选：剔除日均成交金额排名后20%的股票
    amt_threshold = daily_avg_amt['amt'].quantile(0.2)
    selected_stocks = daily_avg_amt[daily_avg_amt['amt'] > amt_threshold]
    # 2.2 合并行业信息并剔除金融、房地产
    selected_stocks = selected_stocks.merge(stock_list[['ts_code', 'industry']], on="ts_code", how="left")
    selected_stocks = selected_stocks[~selected_stocks['industry'].isin(['金融', '房地产'])]
    
    # 2.3 ROE稳定性筛选
    if not roe_data.empty:
        roe_rank_threshold = roe_data['roe_std'].quantile(0.9)
        roe_selected = roe_data[roe_data['roe_std'] < roe_rank_threshold]
    else:
        roe_selected = pd.DataFrame()
    
    # 2.4 财务健康度筛选
    cashflow_selected = cashflow_data[
        (cashflow_data['free_cashflow'] > 0) &
        (cashflow_data['ev'] > 0) &
        (cashflow_data['n_cashflow_act'] > 0)
    ]
    # 2.5 剔除经营现金流占营业利润排名后30%的股票
    cash_to_op_income_threshold = cashflow_selected['cash_to_op_income'].quantile(0.3)
    cashflow_selected = cashflow_selected[cashflow_selected['cash_to_op_income'] > cash_to_op_income_threshold]
    
    # 3. 排序并选股
    final_candidates = cashflow_selected[['ts_code', 'fcf_rate']].dropna()
    final_candidates = final_candidates.sort_values(by='fcf_rate', ascending=False)
    final_index_components = final_candidates.head(10)  # 修改此处，由原来的 head(100) 改为 head(10)
    return final_index_components

def get_next_trading_day(date_str):
    """
    根据给定日期返回下个交易日（使用Tushare交易日历）
    """
    end_date = (pd.to_datetime(date_str) + pd.Timedelta(days=10)).strftime('%Y%m%d')
    cal_df = pro.trade_cal(exchange='SSE', is_open='1', start_date=date_str, end_date=end_date, fields='cal_date')
    if not cal_df.empty:
        return cal_df.iloc[0]['cal_date']
    else:
        return date_str

def get_rebalancing_dates(start_date, end_date):
    """
    根据中国A股财报披露截止日（示例中使用：4月30、8月31、5月15、11月14）生成调仓日期，
    调仓日取截止日后一天并调整为下一个交易日。
    """
    start_year = int(start_date[:4])
    end_year = int(end_date[:4])
    rebalancing_dates = []
    for year in range(start_year, end_year + 1):
        for d in ['0430', '0831', '0515', '1114']:
            date_candidate = f"{year}{d}"
            # 调仓日：披露截止后一天，再调整为下一个交易日
            next_day = (pd.to_datetime(date_candidate) + pd.Timedelta(days=1)).strftime('%Y%m%d')
            trade_day = get_next_trading_day(next_day)
            if start_date <= trade_day <= end_date:
                rebalancing_dates.append(trade_day)
    rebalancing_dates.sort()
    return rebalancing_dates

# 新增函数：获取分红数据并计算分红收益率
def get_dividends(ts_code, start_date, end_date):
    """
    使用 Tushare 获取指定股票在持仓期间的分红数据，计算分红收益率。
    假设字段 'div_cash' 表示每股现金分红，返回的总分红需要除以初始价格来得到收益率。
    """
    try:
        div_df = pro.dividend(ts_code=ts_code, start_date=start_date, end_date=end_date,
                               fields='ts_code, ex_date, div_cash')
        if div_df.empty:
            return 0.0
        # 此处简化处理：将所有分红累加后返回
        div_total = div_df['div_cash'].sum()
        return div_total
    except Exception as e:
        print(f"获取 {ts_code} 分红数据失败: {e}")
        return 0.0

# 修改函数：计算组合收益时考虑分红再投资（采用复合收益因子）
def compute_portfolio_return(selected, start_date, end_date):
    """
    对选出股票从 start_date 到 end_date 的持仓期间计算等权组合总收益，
    计算逻辑：对于每只股票，收益因子 = (收盘价 + 持仓期内累计分红) / 起始价，
    然后组合收益率为各股票收益因子均值减 1。
    """
    factors = []
    for code in selected['ts_code']:
        try:
            df = pro.daily(ts_code=code, start_date=start_date, end_date=end_date, fields='trade_date, close')
            if df.empty:
                continue
            df = df.sort_values('trade_date')
            start_price = df.iloc[0]['close']
            end_price = df.iloc[-1]['close']
            # 获取持仓期间累计分红
            div_total = get_dividends(code, start_date, end_date)
            # 计算复合因子：将分红再投资（直接加到最终价格）
            factor = (end_price + div_total) / start_price
            factors.append(factor)
        except Exception as e:
            print(f"Error fetching data for {code}: {e}")
    if factors:
        avg_factor = np.mean(factors)
        return avg_factor - 1
    else:
        return 0.0

import yfinance as yf
# 修改函数：使用除权价格计算标普收益（反映分红再投资）
def get_benchmark_return_yf(ticker, start_date, end_date):
    """
    使用 yfinance 获取 benchmark（例如标普500）的收益率，
    优先使用 'Adj Close'（除权价格，体现分红再投资后效果）。
    """
    data = yf.download(ticker, start=pd.to_datetime(start_date).strftime('%Y-%m-%d'),
                             end=pd.to_datetime(end_date).strftime('%Y-%m-%d'))
    if data.empty:
        return 0.0
    if 'Adj Close' in data.columns:
        start_price = data.iloc[0]['Adj Close']
        end_price = data.iloc[-1]['Adj Close']
    else:
        start_price = data.iloc[0]['Close']
        end_price = data.iloc[-1]['Close']
    return (end_price / start_price) - 1

def get_benchmark_return_ts(ts_code, start_date, end_date):
    """
    使用 Tushare 获取 benchmark（如沪深300）的收益率
    """
    df = pro.index_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
    if df.empty:
        return 0.0
    df = df.sort_values('trade_date')
    start_price = df.iloc[0]['close']
    end_price = df.iloc[-1]['close']
    return (end_price / start_price) - 1

# ---------------- 回测主程序 ----------------
if __name__ == '__main__':
    # 定义回测起止日期：统一使用TIME_RANGE_START和TIME_RANGE_END
    start_date = TIME_RANGE_START
    end_date   = TIME_RANGE_END
    rebal_dates = get_rebalancing_dates(start_date, end_date)
    print("调仓日期：", rebal_dates)
    
    portfolio_returns = []
    period_dates = []  # 用于绘图横轴
    for i in range(len(rebal_dates) - 1):
        current_date = rebal_dates[i]
        next_date = rebal_dates[i + 1]
        # 选股：基于当前调仓日之前已有的数据（demo中未做严格截止日过滤）
        selected = select_stocks(current_date)
        period_ret = compute_portfolio_return(selected, current_date, next_date)
        portfolio_returns.append(period_ret)
        period_dates.append(current_date)
        print(f"调仓周期 {current_date} 到 {next_date}：收益率 = {period_ret:.2%}")
    
    # 计算累计收益率
    cumulative_return = np.cumprod([1 + r for r in portfolio_returns]) - 1
    
    # Benchmark收益率计算
    sp500_returns = []
    csi300_returns = []
    for i in range(len(rebal_dates) - 1):
        current_date = rebal_dates[i]
        next_date = rebal_dates[i + 1]
        sp500_returns.append(get_benchmark_return_yf('^GSPC', current_date, next_date))
        csi300_returns.append(get_benchmark_return_ts('000300.SH', current_date, next_date))
    sp500_cum = np.cumprod([1 + r for r in sp500_returns]) - 1
    csi300_cum = np.cumprod([1 + r for r in csi300_returns]) - 1
    
# 修改图表部分：配置中文字体为 SimHei，若不可用则退化到英文
import matplotlib.pyplot as plt
import matplotlib
try:
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 尝试使用SimHei中文字体
except Exception as e:
    print("SimHei 字体不可用，使用 默认字体，并将标签调整为英文")
    matplotlib.rcParams['font.sans-serif'] = ['Arial']
    
plt.figure(figsize=(10, 6))
plt.plot(period_dates, cumulative_return, label='Strategy')
plt.plot(period_dates, sp500_cum, label='S&P500')
plt.plot(period_dates, csi300_cum, label='CSI300')
plt.xlabel('Rebalance Date')
plt.ylabel('Cumulative Return')
plt.title('Portfolio vs. Benchmark Returns')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
    
    # 原结果输出与保存（若需要仍可保留）
try:
    import ace_tools as tools
    tools.display_dataframe_to_user(name="精选股票名单", dataframe=selected)
except Exception as e:
    print("ace_tools 模块不可用，直接输出结果：")
    print(selected)

drive.mount('/content/drive')
selected.to_csv("/content/drive/MyDrive/final_index_components.csv", index=False)
print("结果已保存到 Google Drive")