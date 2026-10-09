import streamlit as st

# 1. 强制全局页面配置 (必须放在最前面)
st.set_page_config(
    page_title="预测模型训练与应用平台",
    page_icon="📊",
    layout="wide"
)

# 2. 定义页面路由 (强制改名：左边是真实的物理文件名，title是侧边栏显示的完美中文名)
# 注意：把左侧的字符串换成你电脑里实际的文件名！
page_home = st.Page("home.py", title="首页", icon="🏠")
page_train = st.Page("pages/1_模型构建与验证.py", title="模型训练", icon="🎓")
page_apply = st.Page("pages/2_独立验证与解释分析.py", title="模型应用", icon="🔍")

# 3. 注册并运行导航菜单
pg = st.navigation([page_home, page_train, page_apply])
pg.run()
