# 游戏销量预测系统 - 月度数据专业版
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.statespace.sarimax import SARIMAX
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import io
import warnings
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

# 设置页面
st.set_page_config(
    page_title="游戏销量预测系统 - 月度专业版",
    page_icon="📊",
    layout="wide"
)

# 标题和说明
st.title("📊 游戏销量预测系统 - 月度专业版")
st.markdown("""
本系统专为**月度销售预测**设计，支持深度整合Steam口碑数据，并为国内平台提供替代指标方案。
请按流程操作：**设置时间范围 → 下载模板 → 填写数据 → 上传数据 → 选择预测模式 → 获取预测结果**
""")

# 初始化session state
for key in ['df', 'df_monthly', 'trained_model', 'future_features', 'full_data_with_revenue', 'historical_unit_price']:
    if key not in st.session_state:
        st.session_state[key] = None

# 侧边栏 - 数据输入和参数设置
with st.sidebar:
    st.header("📅 时间范围设置")
    
    # 动态模板生成设置
    col1, col2 = st.columns(2)
    with col1:
        game_release_date = st.date_input(
            "游戏上市日期", 
            value=pd.to_datetime("2023-01-01"),
            help="请选择游戏正式上市的日期",
            key="release_date"
        )
    
    with col2:
        evaluation_base_date = st.date_input(
            "评估基准日期", 
            value=pd.to_datetime("2025-12-31"),
            help="请选择历史数据截止的评估基准日期",
            key="base_date"
        )
    
    # 检查日期有效性
    if evaluation_base_date <= game_release_date:
        st.error("评估基准日期必须晚于游戏上市日期")
    else:
        # 计算月数
        months_diff = (evaluation_base_date.year - game_release_date.year) * 12 + (evaluation_base_date.month - game_release_date.month)
        if evaluation_base_date.day > game_release_date.day:
            months_diff += 1
        
        st.info(f"时间范围: {months_diff}个月 ({game_release_date} 至 {evaluation_base_date})")
    
    # 数据模板下载
    st.markdown("### 📥 下载月度数据模板")
    
    if st.button("生成并下载月度数据模板", type="primary", use_container_width=True):
        if evaluation_base_date <= game_release_date:
            st.error("请先设置正确的日期范围")
        else:
            # 生成月度日期序列
            dates = pd.date_range(start=game_release_date.replace(day=1), 
                                 end=evaluation_base_date.replace(day=1), 
                                 freq='MS')  # 每月第一天
            
            # 创建空白模板（中文表头，无模拟数据）
            template_data = {
                '日期': dates.strftime('%Y-%m-%d'),
                '月销量': [0] * len(dates),
                '单价_元': [35] * len(dates),
                'Steam好评数': [0] * len(dates),
                'Steam差评数': [0] * len(dates),
                '营销事件': [0] * len(dates)
            }
            
            template_df = pd.DataFrame(template_data)
            
            # 计算衍生指标（演示用，用户无需填写）
            template_df['Steam总评论数'] = template_df['Steam好评数'] + template_df['Steam差评数']
            template_df['月度好评率'] = template_df.apply(
                lambda x: x['Steam好评数']/x['Steam总评论数'] if x['Steam总评论数'] > 0 else 0, axis=1)
            template_df['净好评数'] = template_df['Steam好评数'] - template_df['Steam差评数']
            template_df['总销售额_元'] = template_df['月销量'] * template_df['单价_元']
            
            # 提供模板下载
            template_csv = template_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="确认下载月度模板(CSV)",
                data=template_csv,
                file_name=f"游戏月度销量数据模板_{game_release_date}_{evaluation_base_date}.csv",
                mime="text/csv",
                key="download_monthly_template"
            )
    
    st.divider()
    
    # 数据上传
    st.header("📤 数据上传")
    
    # 主销售数据文件
    st.subheader("1. 主销售数据")
    main_file = st.file_uploader("上传月度销售数据文件", type=['xlsx', 'xls', 'csv'], 
                                 help="请使用上方下载的模板格式填写")
    
    # 国内平台替代指标（可选）
    st.subheader("2. 国内平台数据（可选）")
    domestic_file = st.file_uploader("上传国内平台指标文件", type=['xlsx', 'xls', 'csv'],
                                    help="可包含TapTap关注数、B站播放量等替代指标，需有'日期'列")
    
    # 未来营销计划（可选）
    st.subheader("3. 未来营销计划（可选）")
    marketing_file = st.file_uploader("上传未来营销计划", type=['xlsx', 'xls', 'csv'],
                                     help="用于SARIMAX预测，需包含未来日期的营销活动强度")
    
    # 单价设置
    st.subheader("💰 价格设置")
    historical_unit_price = st.number_input(
        "历史数据单价(元)", 
        min_value=1, max_value=500, value=35, step=1,
        help="历史数据中每份游戏的销售单价",
        key="hist_price_input"
    )
    st.session_state.historical_unit_price = historical_unit_price
    
    future_unit_price = st.number_input(
        "未来预测期单价(元)", 
        min_value=1, max_value=500, value=35, step=1,
        help="未来预测期间每份游戏的销售单价"
    )
    
    st.divider()
    
    # 预测模式选择
    st.header("🎯 预测模式设置")
    
    prediction_mode = st.selectbox(
        "选择预测模式",
        ["基础长期预测 (模式A)", "分阶段滚动预测 (模式B)", "SARIMAX高级预测 (模式C)"],
        help="""模式A：直接预测未来多个月份，适合趋势稳定的场景。
               模式B：分阶段滚动更新，预测更贴近最新市场变化。
               模式C：考虑季节性和已知营销计划，预测更精细。"""
    )
    
    # 通用预测设置
    total_forecast_months = st.number_input(
        "预测总月数", 
        min_value=3, max_value=60, value=36, step=3,
        help="设置需要预测的未来月数（最多60个月，即5年）"
    )
    
    # 模式B专属参数
    if prediction_mode == "分阶段滚动预测 (模式B)":
        st.subheader("滚动预测参数")
        rolling_window = st.number_input(
            "滚动窗口月数", 
            min_value=6, max_value=36, value=12,
            help="每次用于重新训练模型的历史数据长度（月）。"
        )
        update_frequency = st.number_input(
            "数据更新频率（月）", 
            min_value=1, max_value=12, value=3,
            help="每积累多少月新数据后，重新运行一次滚动预测。"
        )
    
    # 模式C专属参数
    elif prediction_mode == "SARIMAX高级预测 (模式C)":
        st.subheader("SARIMAX模型参数")
        seasonal_period = st.number_input(
            "季节性周期 (s)", 
            min_value=3, max_value=24, value=12,
            help="对于月度数据，通常设为12（一年）。"
        )
        
        # SARIMAX模型阶数设置
        col1, col2, col3 = st.columns(3)
        with col1:
            p_value = st.number_input("p (自回归)", 0, 5, 1)
        with col2:
            d_value = st.number_input("d (差分)", 0, 2, 1)
        with col3:
            q_value = st.number_input("q (移动平均)", 0, 5, 1)
    
    st.divider()
    
    # 数据处理选项
    st.header("⚙️ 数据处理选项")
    
    use_steam_features = st.checkbox(
        "使用Steam口碑特征", 
        value=True,
        help="使用Steam好评率、净好评数等作为预测特征"
    )
    
    # 自动聚合选项（如果用户上传了周数据）
    auto_aggregate = st.checkbox(
        "自动聚合周数据为月数据", 
        value=False,
        help="如果上传的是周度数据，系统将自动聚合为月度数据"
    )

# 数据处理函数
def process_uploaded_data(main_file, domestic_file, auto_aggregate=False):
    """处理上传的数据文件"""
    if main_file is None:
        return None, "请先上传主销售数据文件"
    
    try:
        # 读取主文件
        if main_file.name.endswith('.csv'):
            df = pd.read_csv(main_file)
        else:
            df = pd.read_excel(main_file)
        
        # 检查必要列
        required_cols = ['日期', '月销量']
        if not all(col in df.columns for col in required_cols):
            return None, f"数据必须包含以下列: {required_cols}"
        
        # 转换日期格式
        df['日期'] = pd.to_datetime(df['日期'])
        
        # 如果是周数据且需要聚合
        if auto_aggregate and '周销量' in df.columns:
            # 假设有周销量和日期列
            df['日期'] = pd.to_datetime(df['日期'])
            df.set_index('日期', inplace=True)
            df_monthly = df.resample('M').agg({
                '周销量': 'sum',
                '单价_元': 'mean' if '单价_元' in df.columns else 'first'
            }).reset_index()
            df_monthly.rename(columns={'周销量': '月销量'}, inplace=True)
            df = df_monthly
            st.success("已自动将周数据聚合为月数据")
        
        # 重命名列名到英文，便于后续处理
        column_mapping = {
            '日期': 'date',
            '月销量': 'monthly_sales',
            '单价_元': 'unit_price',
            'Steam好评数': 'steam_positive',
            'Steam差评数': 'steam_negative',
            'Steam总评论数': 'steam_total',
            '月度好评率': 'positive_rate',
            '净好评数': 'net_positive',
            '营销事件': 'marketing_event',
            '总销售额_元': 'total_revenue'
        }
        
        # 只映射存在的列
        existing_mapping = {k: v for k, v in column_mapping.items() if k in df.columns}
        df = df.rename(columns=existing_mapping)
        
        # 确保有unit_price列
        if 'unit_price' not in df.columns:
            df['unit_price'] = st.session_state.historical_unit_price
            st.info(f"数据中未找到单价列，已使用默认单价 {st.session_state.historical_unit_price} 元")
        
        # 计算Steam特征（如果提供了好评/差评数据）
        if 'steam_positive' in df.columns and 'steam_negative' in df.columns:
            df['steam_total'] = df['steam_positive'] + df['steam_negative']
            df['positive_rate'] = df.apply(
                lambda x: x['steam_positive']/x['steam_total'] if x['steam_total'] > 0 else 0, 
                axis=1
            )
            df['net_positive'] = df['steam_positive'] - df['steam_negative']
        
        # 计算总销售额
        df['total_revenue'] = df['monthly_sales'] * df['unit_price']
        
        # 处理国内平台数据
        domestic_features = None
        if domestic_file is not None:
            try:
                if domestic_file.name.endswith('.csv'):
                    domestic_df = pd.read_csv(domestic_file)
                else:
                    domestic_df = pd.read_excel(domestic_file)
                
                if '日期' in domestic_df.columns:
                    domestic_df['日期'] = pd.to_datetime(domestic_df['日期'])
                    domestic_df.set_index('日期', inplace=True)
                    
                    # 如果是高频数据，聚合为月度
                    if len(domestic_df) > len(df) * 4:  # 粗略判断为高频数据
                        domestic_df = domestic_df.resample('M').mean()
                    
                    domestic_df.index.name = 'date'
                    domestic_features = domestic_df
                    st.success(f"已成功加载国内平台指标，共{len(domestic_df)}个月度数据点")
            except Exception as e:
                st.warning(f"国内平台数据读取失败，将继续使用其他特征: {str(e)}")
        
        # 处理未来营销计划
        future_marketing = None
        if marketing_file is not None and prediction_mode == "SARIMAX高级预测 (模式C)":
            try:
                if marketing_file.name.endswith('.csv'):
                    marketing_df = pd.read_csv(marketing_file)
                else:
                    marketing_df = pd.read_excel(marketing_file)
                
                if '日期' in marketing_df.columns:
                    marketing_df['日期'] = pd.to_datetime(marketing_df['日期'])
                    marketing_df.set_index('日期', inplace=True)
                    marketing_df.index.name = 'date'
                    future_marketing = marketing_df
                    st.success("已成功加载未来营销计划")
            except Exception as e:
                st.warning(f"营销计划数据读取失败: {str(e)}")
        
        # 添加时间特征
        df = df.sort_values('date')
        df['month_num'] = range(1, len(df) + 1)
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        
        # 合并国内平台特征
        if domestic_features is not None:
            df = df.merge(domestic_features, left_on='date', right_index=True, how='left')
        
        return df, "数据加载成功"
        
    except Exception as e:
        return None, f"数据处理失败: {str(e)}"

# 预测模型函数
def train_sarimax_model(data, target_col='monthly_sales', exogenous_cols=None, order=(1,1,1), seasonal_order=(1,1,1,12)):
    """训练SARIMAX模型"""
    if exogenous_cols is None:
        exogenous_cols = []
    
    # 准备数据
    y = data[target_col]
    X = data[exogenous_cols] if exogenous_cols else None
    
    # 训练模型
    model = SARIMAX(
        y, 
        exog=X,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    
    results = model.fit(disp=False)
    return results

def rolling_forecast(data, target_col, total_months, window_months, update_freq, feature_cols=None):
    """分阶段滚动预测"""
    if feature_cols is None:
        feature_cols = []
    
    all_predictions = []
    all_dates = []
    
    # 确保数据按时间排序
    data = data.sort_values('date').copy()
    
    # 生成未来日期
    last_date = data['date'].iloc[-1]
    future_dates = pd.date_range(
        last_date + pd.offsets.MonthBegin(1), 
        periods=total_months, 
        freq='MS'
    )
    
    current_data = data.copy()
    
    for start_idx in range(0, total_months, update_freq):
        # 当前阶段的预测月数
        steps = min(update_freq, total_months - start_idx)
        
        # 使用最近window_months个月的数据
        if len(current_data) > window_months:
            train_data = current_data.iloc[-window_months:].copy()
        else:
            train_data = current_data.copy()
        
        # 准备外生变量
        exogenous_cols = [col for col in feature_cols if col in train_data.columns]
        X_train = train_data[exogenous_cols] if exogenous_cols else None
        
        # 训练模型（简化为非季节性ARIMA）
        model = SARIMAX(
            train_data[target_col],
            exog=X_train,
            order=(1,1,1),
            seasonal_order=(0,0,0,0),
            enforce_stationarity=False
        )
        results = model.fit(disp=False)
        
        # 为未来期间准备外生变量（这里简化处理）
        X_future = None
        if exogenous_cols:
            # 在实际应用中，这里需要用户提供或合理推测未来的外生变量值
            X_future = pd.DataFrame(
                np.zeros((steps, len(exogenous_cols))),
                columns=exogenous_cols
            )
        
        # 进行预测
        forecast = results.get_forecast(steps=steps, exog=X_future)
        pred_mean = forecast.predicted_mean
        pred_conf = forecast.conf_int()
        
        # 记录预测结果
        pred_dates = future_dates[start_idx:start_idx+steps]
        for i, date in enumerate(pred_dates):
            all_predictions.append({
                'date': date,
                'predicted_sales': pred_mean.iloc[i] if i < len(pred_mean) else pred_mean.iloc[-1],
                'lower_bound': pred_conf.iloc[i, 0] if i < len(pred_conf) else pred_conf.iloc[-1, 0],
                'upper_bound': pred_conf.iloc[i, 1] if i < len(pred_conf) else pred_conf.iloc[-1, 1]
            })
            all_dates.append(date)
        
        # 更新当前数据（在实际应用中，这里会添加真实的新数据）
        # 这里我们简单地将预测值作为"新数据"加入以进行下一轮滚动（仅演示用）
        if start_idx + update_freq < total_months:
            new_data = pd.DataFrame({
                'date': pred_dates,
                target_col: pred_mean.values,
                'unit_price': [future_unit_price] * len(pred_dates)
            })
            # 为其他特征填充默认值
            for col in feature_cols:
                if col not in new_data.columns and col in current_data.columns:
                    new_data[col] = current_data[col].iloc[-1] if len(current_data) > 0 else 0
            
            current_data = pd.concat([current_data, new_data], ignore_index=True)
            current_data = current_data.sort_values('date').tail(window_months + steps)
    
    predictions_df = pd.DataFrame(all_predictions)
    return predictions_df

# 主程序逻辑
if main_file is not None:
    # 处理上传的数据
    with st.spinner("正在处理上传的数据..."):
        df, message = process_uploaded_data(main_file, domestic_file, auto_aggregate)
        
        if df is not None:
            st.session_state.df = df
            st.success(f"{message}! 共 {len(df)} 个月度数据点")
            
            # 显示数据预览
            st.header("📋 数据预览")
            
            # 转换回中文显示
            display_df = df.copy()
            chinese_columns = {
                'date': '日期',
                'monthly_sales': '月销量',
                'unit_price': '单价_元',
                'total_revenue': '总销售额_元',
                'steam_positive': 'Steam好评数',
                'steam_negative': 'Steam差评数',
                'positive_rate': '月度好评率',
                'net_positive': '净好评数',
                'marketing_event': '营销事件'
            }
            
            existing_mapping = {k: v for k, v in chinese_columns.items() if k in display_df.columns}
            display_df = display_df.rename(columns=existing_mapping)
            
            # 显示数据
            col1, col2 = st.columns([3, 1])
            with col1:
                # 只显示主要列
                display_cols = ['日期', '月销量', '单价_元', '总销售额_元']
                if '月度好评率' in display_df.columns:
                    display_cols.append('月度好评率')
                if '营销事件' in display_df.columns:
                    display_cols.append('营销事件')
                
                available_cols = [col for col in display_cols if col in display_df.columns]
                st.dataframe(display_df[available_cols], use_container_width=True)
            
            with col2:
                # 显示关键指标
                total_sales = df['monthly_sales'].sum()
                avg_monthly_sales = df['monthly_sales'].mean()
                total_revenue = df['total_revenue'].sum()
                
                st.metric("总月数", len(df))
                st.metric("平均月销量", f"{int(avg_monthly_sales):,}")
                st.metric("历史总销量", f"{total_sales:,}")
                st.metric("历史总销售额", f"¥{total_revenue:,.0f}")
                
                if 'positive_rate' in df.columns:
                    avg_positive_rate = df['positive_rate'].mean()
                    st.metric("平均好评率", f"{avg_positive_rate:.1%}")
            
            # 数据可视化
            st.header("📈 数据可视化分析")
            
            tab1, tab2, tab3 = st.tabs(["销量趋势", "口碑分析", "财务指标"])
            
            with tab1:
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(x=df['date'], y=df['monthly_sales'], 
                                         mode='lines+markers', name='月销量',
                                         line=dict(color='blue', width=2)))
                
                if 'marketing_event' in df.columns:
                    event_dates = df[df['marketing_event'] > 0]['date']
                    event_sales = df[df['marketing_event'] > 0]['monthly_sales']
                    fig1.add_trace(go.Scatter(x=event_dates, y=event_sales,
                                             mode='markers', name='营销事件',
                                             marker=dict(color='red', size=10, symbol='diamond')))
                
                fig1.update_layout(title="月销量趋势与营销事件", 
                                 xaxis_title="日期", 
                                 yaxis_title="销量")
                st.plotly_chart(fig1, use_container_width=True)
            
            with tab2:
                if 'positive_rate' in df.columns:
                    fig2 = make_subplots(
                        rows=2, cols=2,
                        subplot_titles=("月度好评率趋势", "Steam评论数", "好评vs差评", "净好评数趋势"),
                        vertical_spacing=0.15
                    )
                    
                    # 月度好评率
                    fig2.add_trace(
                        go.Scatter(x=df['date'], y=df['positive_rate'], 
                                  mode='lines+markers', name='好评率',
                                  line=dict(color='green')),
                        row=1, col=1
                    )
                    
                    # Steam评论数
                    if 'steam_total' in df.columns:
                        fig2.add_trace(
                            go.Scatter(x=df['date'], y=df['steam_total'], 
                                      mode='lines', name='总评论数',
                                      line=dict(color='blue')),
                            row=1, col=2
                        )
                    
                    # 好评vs差评
                    if 'steam_positive' in df.columns and 'steam_negative' in df.columns:
                        fig2.add_trace(
                            go.Bar(x=df['date'], y=df['steam_positive'], name='好评数'),
                            row=2, col=1
                        )
                        fig2.add_trace(
                            go.Bar(x=df['date'], y=df['steam_negative'], name='差评数'),
                            row=2, col=1
                        )
                    
                    # 净好评数
                    if 'net_positive' in df.columns:
                        fig2.add_trace(
                            go.Scatter(x=df['date'], y=df['net_positive'], 
                                      mode='lines', name='净好评数',
                                      line=dict(color='orange')),
                            row=2, col=2
                        )
                    
                    fig2.update_layout(height=600, showlegend=True, barmode='stack')
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.info("未找到Steam口碑数据，无法生成口碑分析图表")
            
            with tab3:
                fig3 = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=("月销量趋势", "月销售额趋势", "单价趋势", "累计销售额"),
                    vertical_spacing=0.15
                )
                
                # 月销量
                fig3.add_trace(
                    go.Scatter(x=df['date'], y=df['monthly_sales'], mode='lines', name='月销量'),
                    row=1, col=1
                )
                
                # 月销售额
                fig3.add_trace(
                    go.Scatter(x=df['date'], y=df['total_revenue'], mode='lines', name='月销售额', 
                              line=dict(color='green')),
                    row=1, col=2
                )
                
                # 单价趋势
                fig3.add_trace(
                    go.Scatter(x=df['date'], y=df['unit_price'], mode='lines+markers', name='单价', 
                              line=dict(color='orange')),
                    row=2, col=1
                )
                
                # 累计销售额
                cumulative_revenue = df['total_revenue'].cumsum()
                fig3.add_trace(
                    go.Scatter(x=df['date'], y=cumulative_revenue, mode='lines', name='累计销售额', 
                              line=dict(color='purple')),
                    row=2, col=2
                )
                
                fig3.update_layout(height=600, showlegend=True)
                st.plotly_chart(fig3, use_container_width=True)
            
            # 模型训练部分
            st.header("🤖 模型训练与预测")
            
            if st.button("开始训练模型并预测", type="primary", use_container_width=True):
                with st.spinner("正在训练模型并进行预测..."):
                    try:
                        # 准备特征列
                        feature_cols = ['month_num']
                        
                        # 添加Steam口碑特征（如果可用且用户选择使用）
                        if use_steam_features:
                            steam_features = ['positive_rate', 'net_positive', 'steam_total']
                            for feat in steam_features:
                                if feat in df.columns:
                                    feature_cols.append(feat)
                        
                        # 添加其他特征
                        other_features = ['marketing_event']
                        for feat in other_features:
                            if feat in df.columns:
                                feature_cols.append(feat)
                        
                        # 添加国内平台特征（动态识别）
                        domestic_feature_prefixes = ['taptap', 'bilibili', 'weibo', 'index']
                        for col in df.columns:
                            if any(prefix in col.lower() for prefix in domestic_feature_prefixes):
                                feature_cols.append(col)
                        
                        # 确保特征列存在
                        for col in feature_cols:
                            if col not in df.columns:
                                df[col] = 0
                        
                        # 根据选择的模式进行预测
                        if prediction_mode == "基础长期预测 (模式A)":
                            st.subheader("📈 基础长期预测结果")
                            
                            # 使用SARIMAX模型
                            model_results = train_sarimax_model(
                                df, 
                                target_col='monthly_sales',
                                exogenous_cols=[col for col in feature_cols if col != 'month_num'],
                                order=(1,1,1),
                                seasonal_order=(1,1,1,12)
                            )
                            
                            # 准备未来外生变量
                            X_future = None
                            exogenous_for_future = [col for col in feature_cols if col != 'month_num']
                            if exogenous_for_future:
                                last_values = df[exogenous_for_future].iloc[-1:].copy()
                                X_future = pd.concat([last_values] * total_forecast_months, ignore_index=True)
                            
                            # 进行预测
                            forecast = model_results.get_forecast(
                                steps=total_forecast_months,
                                exog=X_future
                            )
                            
                            pred_mean = forecast.predicted_mean
                            pred_conf = forecast.conf_int()
                            
                            # 生成未来日期
                            last_date = df['date'].iloc[-1]
                            future_dates = pd.date_range(
                                last_date + pd.offsets.MonthBegin(1), 
                                periods=total_forecast_months, 
                                freq='MS'
                            )
                            
                            # 创建预测结果DataFrame
                            predictions = pd.DataFrame({
                                'date': future_dates,
                                'predicted_sales': pred_mean.values,
                                'lower_bound': pred_conf.iloc[:, 0].values,
                                'upper_bound': pred_conf.iloc[:, 1].values,
                                'unit_price': future_unit_price
                            })
                            
                            predictions['predicted_revenue'] = predictions['predicted_sales'] * predictions['unit_price']
                            
                        elif prediction_mode == "分阶段滚动预测 (模式B)":
                            st.subheader("🔄 分阶段滚动预测结果")
                            
                            predictions = rolling_forecast(
                                df,
                                target_col='monthly_sales',
                                total_months=total_forecast_months,
                                window_months=rolling_window,
                                update_freq=update_frequency,
                                feature_cols=[col for col in feature_cols if col != 'month_num']
                            )
                            
                            predictions['unit_price'] = future_unit_price
                            predictions['predicted_revenue'] = predictions['predicted_sales'] * predictions['unit_price']
                            
                        elif prediction_mode == "SARIMAX高级预测 (模式C)":
                            st.subheader("🎯 SARIMAX高级预测结果")
                            
                            # 使用用户设置的参数
                            model_results = train_sarimax_model(
                                df, 
                                target_col='monthly_sales',
                                exogenous_cols=[col for col in feature_cols if col != 'month_num'],
                                order=(p_value, d_value, q_value),
                                seasonal_order=(1,1,1,seasonal_period)
                            )
                            
                            # 如果有未来的营销计划，使用它作为外生变量
                            X_future = None
                            if marketing_file is not None:
                                # 这里应该使用之前加载的future_marketing数据
                                # 简化处理：使用最后的值
                                exogenous_for_future = [col for col in feature_cols if col != 'month_num']
                                if exogenous_for_future:
                                    last_values = df[exogenous_for_future].iloc[-1:].copy()
                                    X_future = pd.concat([last_values] * total_forecast_months, ignore_index=True)
                            
                            # 进行预测
                            forecast = model_results.get_forecast(
                                steps=total_forecast_months,
                                exog=X_future
                            )
                            
                            pred_mean = forecast.predicted_mean
                            pred_conf = forecast.conf_int()
                            
                            # 生成未来日期
                            last_date = df['date'].iloc[-1]
                            future_dates = pd.date_range(
                                last_date + pd.offsets.MonthBegin(1), 
                                periods=total_forecast_months, 
                                freq='MS'
                            )
                            
                            predictions = pd.DataFrame({
                                'date': future_dates,
                                'predicted_sales': pred_mean.values,
                                'lower_bound': pred_conf.iloc[:, 0].values,
                                'upper_bound': pred_conf.iloc[:, 1].values,
                                'unit_price': future_unit_price
                            })
                            
                            predictions['predicted_revenue'] = predictions['predicted_sales'] * predictions['unit_price']
                        
                        # 保存预测结果
                        st.session_state.future_features = predictions
                        
                        # 显示预测结果
                        st.success("预测完成！")
                        
                        # 显示预测摘要
                        col1, col2, col3, col4 = st.columns(4)
                        total_pred_sales = predictions['predicted_sales'].sum()
                        total_pred_revenue = predictions['predicted_revenue'].sum()
                        avg_monthly_sales_pred = predictions['predicted_sales'].mean()
                        avg_monthly_revenue_pred = predictions['predicted_revenue'].mean()
                        
                        with col1:
                            st.metric("预测期总销量", f"{total_pred_sales:,.0f}")
                        with col2:
                            st.metric("预测期总销售额", f"¥{total_pred_revenue:,.0f}")
                        with col3:
                            st.metric("平均月销量", f"{avg_monthly_sales_pred:,.0f}")
                        with col4:
                            st.metric("平均月销售额", f"¥{avg_monthly_revenue_pred:,.0f}")
                        
                        # 可视化预测结果
                        fig_pred = go.Figure()
                        
                        # 历史数据
                        fig_pred.add_trace(go.Scatter(
                            x=df['date'], y=df['monthly_sales'],
                            mode='lines', name='历史销量',
                            line=dict(color='blue', width=2)
                        ))
                        
                        # 预测数据
                        fig_pred.add_trace(go.Scatter(
                            x=predictions['date'], y=predictions['predicted_sales'],
                            mode='lines', name='预测销量',
                            line=dict(color='green', width=2, dash='dash')
                        ))
                        
                        # 置信区间
                        fig_pred.add_trace(go.Scatter(
                            x=pd.concat([predictions['date'], predictions['date'][::-1]]),
                            y=pd.concat([predictions['upper_bound'], predictions['lower_bound'][::-1]]),
                            fill='toself', fillcolor='rgba(0, 255, 0, 0.2)',
                            line=dict(color='rgba(255, 255, 255, 0)'),
                            name='置信区间'
                        ))
                        
                        fig_pred.update_layout(
                            title=f"{prediction_mode} - 销量预测结果",
                            xaxis_title="日期",
                            yaxis_title="月销量",
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig_pred, use_container_width=True)
                        
                        # 显示详细预测表
                        st.subheader("📋 详细预测数据")
                        
                        display_predictions = predictions.copy()
                        display_predictions['日期'] = display_predictions['date'].dt.strftime('%Y-%m-%d')
                        display_predictions['预测销量'] = display_predictions['predicted_sales'].round(0)
                        display_predictions['单价_元'] = display_predictions['unit_price']
                        display_predictions['预测销售额_元'] = display_predictions['predicted_revenue'].round(2)
                        
                        display_cols = ['日期', '预测销量', '单价_元', '预测销售额_元']
                        if 'lower_bound' in display_predictions.columns:
                            display_predictions['预测下限'] = display_predictions['lower_bound'].round(0)
                            display_predictions['预测上限'] = display_predictions['upper_bound'].round(0)
                            display_cols.extend(['预测下限', '预测上限'])
                        
                        st.dataframe(display_predictions[display_cols], use_container_width=True)
                        
                        # 数据导出
                        st.subheader("💾 数据导出")
                        
                        # 准备完整数据（历史+预测）
                        historical_display = df[['date', 'monthly_sales', 'unit_price', 'total_revenue']].copy()
                        historical_display['数据类型'] = '历史数据'
                        
                        future_display = predictions[['date', 'unit_price']].copy()
                        future_display['monthly_sales'] = predictions['predicted_sales']
                        future_display['total_revenue'] = predictions['predicted_revenue']
                        future_display['数据类型'] = '预测数据'
                        
                        full_data = pd.concat([historical_display, future_display], ignore_index=True)
                        
                        # 转换为中文
                        full_data_display = full_data.copy()
                        full_data_display = full_data_display.rename(columns={
                            'date': '日期',
                            'monthly_sales': '销量',
                            'unit_price': '单价_元',
                            'total_revenue': '总销售额_元',
                            '数据类型': '数据类型'
                        })
                        
                        full_data_display['序号'] = range(1, len(full_data_display) + 1)
                        full_data_display = full_data_display[['序号', '日期', '数据类型', '销量', '单价_元', '总销售额_元']]
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # CSV导出
                            csv_data = full_data_display.to_csv(index=False, encoding='utf-8-sig')
                            st.download_button(
                                label="下载完整数据(CSV)",
                                data=csv_data,
                                file_name=f"游戏销量预测_{prediction_mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                        
                        with col2:
                            # Excel导出
                            excel_buffer = io.BytesIO()
                            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                                full_data_display.to_excel(writer, sheet_name='完整数据', index=False)
                                
                                # 添加摘要
                                summary_data = {
                                    '预测模式': [prediction_mode],
                                    '预测总月数': [total_forecast_months],
                                    '历史数据量': [len(historical_display)],
                                    '预测数据量': [len(future_display)],
                                    '历史总销量': [historical_display['monthly_sales'].sum()],
                                    '预测总销量': [future_display['monthly_sales'].sum()],
                                    '历史总销售额': [historical_display['total_revenue'].sum()],
                                    '预测总销售额': [future_display['total_revenue'].sum()]
                                }
                                summary_df = pd.DataFrame(summary_data)
                                summary_df.to_excel(writer, sheet_name='预测摘要', index=False)
                            
                            excel_buffer.seek(0)
                            st.download_button(
                                label="下载完整报告(Excel)",
                                data=excel_buffer,
                                file_name=f"游戏销量预测报告_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True
                            )
                        
                    except Exception as e:
                        st.error(f"预测过程中出现错误: {str(e)}")
        else:
            st.error(message)

else:
    # 显示欢迎界面
    st.header("欢迎使用游戏销量预测系统 - 月度专业版")
    
    st.markdown("""
    ### 🚀 使用流程
    
    1.  **设置时间范围**：在左侧边栏设置游戏的上市日期和评估基准日期
    2.  **下载数据模板**：系统会根据您的日期范围生成对应的月度数据模板
    3.  **填写历史数据**：在下载的模板中填写游戏的历史月度销售数据
    4.  **上传数据文件**：上传已填写好的数据文件（支持CSV和Excel格式）
    5.  **选择预测模式**：根据需求选择基础预测、滚动预测或SARIMAX高级预测
    6.  **设置预测参数**：配置预测月数、模型参数等
    7.  **训练与预测**：系统将训练模型并生成未来销量预测
    8.  **导出预测结果**：下载包含历史数据和预测结果的完整报告
    
    ### 📋 月度数据模板说明
    
    系统生成的模板包含以下列（全部使用中文表头）：
    
    | 列名 | 说明 | 填写要求 |
    |------|------|----------|
    | **日期** | 月度销售日期 | 格式：YYYY-MM-01，系统已自动生成 |
    | **月销量** | 该月的总销量 | 填写整数，如：4500 |
    | **单价_元** | 游戏的销售单价 | 填写数值，如：35.00 |
    | **Steam好评数** | Steam平台月新增好评数 | 填写整数，如：120（可选） |
    | **Steam差评数** | Steam平台月新增差评数 | 填写整数，如：15（可选） |
    | **营销事件** | 营销活动类型 | 0=无事件, 1=小型更新, 2=大型活动, 3=节日促销（可选） |
    
    **注意**：系统会自动计算衍生指标（好评率、净好评数、总销售额等）。
    """)
    
    # 功能亮点
    st.divider()
    st.subheader("✨ 系统核心功能")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **📊 月度数据优化**
        - 专为Steam月度评论数据设计
        - 降低数据波动，突出长期趋势
        - 支持周数据自动聚合为月数据
        """)
        
        st.info("""
        **👍 深度口碑分析**
        - 整合Steam好评/差评数据
        - 自动计算好评率、净好评数
        - 口碑指标作为预测特征
        """)
    
    with col2:
        st.info("""
        **🔄 多模式预测**
        - 基础长期预测（模式A）
        - 分阶段滚动预测（模式B）
        - SARIMAX高级预测（模式C）
        - 支持最长5年预测
        """)
        
        st.info("""
        **🌐 国内平台适配**
        - 支持国内平台替代指标
        - 灵活处理不同数据格式
        - 适应多种数据采集场景
        """)

# 页脚
st.divider()
st.markdown("""
<div style='text-align: center; color: gray;'>
    <small>📊 游戏销量预测系统 月度专业版 | 支持Steam口碑深度分析 | 版本 4.0 | 纯中文界面与模板</small>
</div>
""", unsafe_allow_html=True)