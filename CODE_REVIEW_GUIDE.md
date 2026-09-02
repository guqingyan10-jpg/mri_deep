# 新增评价代码审核入口

请在默认 `main` 分支按以下顺序查看：

1. [病灶级核心实现](evaluation/wt_lesion_stratified.py)  
   WT/ET共用的分层、26邻域连通域、一对一匹配、Recall、Miss rate、Matched Dice和GT-anchored Dice。

2. [训练集拟合WT/ET阈值](scripts/derive_train_lesion_strata.py)  
   只使用实际训练集拟合small/medium/large阈值。

3. [WT病灶分层评价](scripts/eval_wt_lesion_stratified.py)  
   默认在固定测试集评价五个模型，也包含显式的探索性`valid_test`模式。

4. [ET病灶分层评价](scripts/eval_et_lesion_stratified.py)  
   复用同一套病灶级评价逻辑，评价区域改为ET。

5. [α敏感性分析](scripts/eval_alpha_sensitivity.py)  
   三个种子的best checkpoint分别评价α=0、learned α和α=1，固定37例测试集。

对应测试：

- [病灶分层与匹配测试](tests/test_wt_lesion_stratified.py)
- [WT/ET入口与模型注册测试](tests/test_wt_eval_model_registry.py)
- [α敏感性测试](tests/test_alpha_sensitivity.py)
