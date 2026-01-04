## 一、 软件质量基本概念 (Software Quality Basics)

- Quality (质量)：一种内在的或辨别性的特征。
- Customer view (客户视角)：Fit for use or meet the needs (适合使用或满足需求)。
- Project Manager view (项目经理视角)：Deliver compliant products in time (准时交付符合规范的产品)。
- Developer/Tester view (开发/测试视角)：Bug-free (无缺陷)。
- McCall’s Quality Model (McCall质量模型)：
  - Product revision (产品修订)：Maintainability (可维护性)、Flexibility (灵活性)、Testability (可测试性)。
  - Product transition (产品转移)：Portability (可移植性)、Reusability (复用性)、Interoperability (互操作性)。
  - Product operations (产品运行)：Correctness (正确性)、Reliability (可靠性)、Efficiency (效率)、Integrity (完整性)、Usability (可用性)。
- Defect Introduction (缺陷引入分布)：50% 来自需求理解不足，30% 来自设计翻译错误，20% 来自编码错误。
- Cost of fixing (修复成本)：修复缺陷的花销随生命周期阶段（需求 -> 设计 -> 编码 -> 测试 -> 运行）推移而剧增。

------

## 二、 质量保证与测试 (Quality Assurance vs. Test)

- **Quality Assurance (质量保证/QA)**：Process/Policy Oriented (过程/政策导向)，旨在 Prevent Defects (预防缺陷)，手段为 Audit/Governance (审计/治理)。
- **Test (测试)**：Products/Deliverables Oriented (产品/交付物导向)，旨在 Discover Defects (发现缺陷)，是质量保证的手段之一。
- **Evaluation (评估维度)**：
  - **Requirements (需求评估)**：使用评分模型。
  - **Design (设计评估)**：概要设计/架构评估（Capacity Planning 容量规划）、详细设计评估（设计模式、代码异味重构）。
  - **Code (代码评估)**：静态测试（代码分析）、Code Smell (代码异味)。
- **Process Improvement (过程改进)**：包括沟通协作、开发方法（敏捷）、自动化工具（CI/CD）、规范化（代码规范）、度量反馈（评价指标）。

------

## 三、 测试分类 (Test Classification)

- **By Methodology (按测试方法划分)**：
  - **White-Box Test (白盒测试)**：语句覆盖、条件覆盖、路径覆盖等。
  - **Black-Box Test (黑盒测试)**：等价类法、边界值法等。
  - **Grey-Box Test (灰盒测试)**：结合黑白盒特征。
- **By Granularity/Scope (按测试粒度或范围)**：
  - **Unit Test (单元测试)**：代码级测试。
  - **Integration Test (集成测试)**：接口与组件间测试。
  - **System Test (系统测试)**：系统级完整测试。
  - **Acceptance Test (验收测试)**：交付前确认。
- **By Standards (按评价标准划分)**：Functional (功能性)、Performance (性能)、Security (安全性)、High Availability (高可用性)。
- **Special Terms (特殊术语)**：
  - **Smoke Test (冒烟测试)**：关联 Entry Criteria/EC1 (准入准则)。
  - **Regression Test (回归测试)**：关联 Exit Criteria/EC2 (准出准则)，防止功能退化。
  - **α Test (α测试)**：开发团队内部测试，涵盖各方面。
  - **β Test (β测试)**：最终用户测试，主要针对功能，采用黑盒法。

------

## 四、 测试流程与开发模型 (Process & Development Models)

- **SDP (Software Development Process，软件开发流程)**：定义时间、人员、任务及提交物。
- **Models (开发模型)**：
  - **Waterfall Model (瀑布模型)**：计划 -> 需求 -> 设计 -> 编码 -> 测试 -> 运行。
  - **V-Model (V模型)**：强调测试与开发的对应关系（如需求对应验收测试，设计对应系统测试）。
  - **UP (Unified Process，统一过程)**：迭代式，包含 Inception (初始)、Elaboration (细化)、Construction (构建)、Transition (交付)。
  - **Agile/Scrum (敏捷/Scrum)**：Sprint (短周期迭代)、Daily Meeting (每日站会)、Backlog (待办事项)。
  - **TDD (Test Driven Development，测试驱动开发)**：测试先行，重构驱动。

------

## 五、 核心产出物 (Core Deliverables)

- **Test Plan (测试计划)**：包括 Schedule (进度)、Source/Team (资源/团队)、Scope (范围)。
- **Test Case (测试用例)**：定义测什么、怎么测。
- **Bug Reports (缺陷报告)**：Excel 记录缺陷详情。
- **Test Report (测试报告)**：PPT 形式，汇报当前位置与后续方案。
- **Internal Logic (内在逻辑)**：
  - **Traceability (可追溯性)**：金字塔结构（需求 -> 特性 -> 用例 -> 场景 -> 测试用例）。
  - **Coverage (覆盖率)**：用例对需求/功能点的覆盖程度。

------

## 六、 实践应用 (Practical Applications)

- **Capacity Planning (容量规划)**：针对 AI 业务（训练、推理、RAG）设计硬件配置。
- **Evaluation Metrics (评估指标)**：如推理性能指标 Tokens/s。
- **Evaluation Report (评估报告)**：评估标准（下限）、模型构建、架构改进方案。

---

## 一、 测试计划定义 (Test Planning Definition)

- **Test Planning (测试计划)**：是一种资源组织方式，是从一个状态经过一系列步骤到达一个目标的过程。
- **Three Elements of Planning (计划三要素)**：
  - **Scope (范围)**：Setting Scope (设定范围/目标)。
  - **Schedule (路径)**：Setting Path/Schedule (设定路径/进度)。
  - **Resource (资源)**：Setting Resource (设定资源组织)。
- **Quality Level Planning & Software Metrics (质量水平计划与软件度量)**：量化计划的执行效果。
- **Admission Criteria / Entry Criteria (AC/EC1, 准入准则)**：进入测试阶段必须满足的条件。
- **Exit Criteria (EC2, 准出准则)**：完成测试阶段必须达到的标准。

## 二、 设定范围 (Setting Scope)

- **Requirement (需求)**：为整个软件开发或某个阶段制定相应的目标。
- **Requirement Engineering (需求工程)**：获取、分析、完善需求，并在 SDP (软件开发流程) 中进行需求变更的方法论。
- **Conditions and Goals (条件与目标)**：两者互为因果，初始条件的变化会导致目标的迭代与偏离。
- **Setting Conditions (设定条件)**：
  - **Business Model (商业模式)**：商业目标与愿景。
  - **Functional/Non-functional Requirements (功能/非功能需求)**：总体的技术需求。
  - **Constraint (约束条件)**：人力资源、基础设施、技术储备、预算、开始时间。
- **Setting Goals (设定目标)**：
  - **Technical Metrics (技术指标)**：功能、质量、技术先进性。
  - **Operational Capability (运维能力)**：可靠性。
  - **Cost & Time (成本与时间)**：预算控制与结束时间。
- **Requirement Change (需求变更)**：实质上是目标的变更，源于实现目标或现实条件与假设的差异。

## 三、 设定路径 (Setting Schedule)

- **Preparation Phase (准备期)**：
  - **Business Modeling (商业模型建立)**。
  - **Requirement Analysis (需求分析)**。
  - **Environment Setup (环境搭建)**：运行环境 (OP)、执行环境 (SA)、支持环境 (Infra)。
  - **Planning (制定计划)**：计划是逐步清晰、逐步细化的。
- **QA Core Work in Requirement Phase (需求阶段 QA 核心工作)**：
  - **Testability (可测试性)**：每个功能是否有合适的测试方法。
  - **Coverage Strategy (覆盖策略)**：如何覆盖功能。
  - **Test Case Inventory (测试用例大纲)**：具体功能采用的测试方法与数据来源。
  - **Baseline Understanding (理解基线)**：团队成员对需求的理解是否一致。
  - **Evaluation & Threshold (评估方法与阈值)**：制定评估准则。

## 四、 组织资源 (Resource Organization)

- **Resource Types (资源类型)**：
  - **Human/Organization Resource (人力/组织资源)**：构建可用的团队。
  - **Physical/Hardware/Software Resource (物理/软硬件资源)**：获得基础设施。
  - **Knowledge Resource (知识资源)**：技术支持与技术储备。
  - **Business Resource (业务资源)**：接触最终用户。
  - **Management Resource (管理资源)**：获得高层重视。
- **Team Roles in Iteration (迭代中的团队角色)**：
  - **BA (Business Analyst, 业务分析师)**：需求分析与文档。
  - **DEV (Developer, 开发人员)**：设计、实现、优化、改 Bug。
  - **QA (Quality Assurance, 质量保证/测试)**：写用例、执行测试、提交报告。
  - **IA (Infrastructure Architect, 基础设施架构师)**：支持环境维护。
  - **PM (Project Manager, 项目经理)**：跟踪进度、协调资源。
  - **OP (Operations, 运维人员)**：部署与优化。

## 五、 软件度量体系 (Software Metrics System)

- **Metric Objectives (度量目标)**：改进质量、控制成本、提高效率。
- **Benchmarking (基准比较)**：与行业标准或组织历史结果对比设定目标。
- **Product Metrics (产品度量)**：
  - **Size Metrics (规模度量)**：LOC (Lines of Code, 代码行数)、FP (Function Points, 功能点)。
  - **Quality Metrics (质量度量)**：Defect Density (缺陷密度)、Reliability (可靠性 - 如 MTTF 平均失效前时间)、Maintainability (维护性)、Testability (可测试性 - 如自动化覆盖率)。
  - **Performance/Security Metrics (性能/安全性度量)**。
- **Process Metrics (过程度量)**：
  - **Progress Metrics (进度度量)**：Schedule Variance (计划偏差)、Critical Path Completion Rate (关键路径完成率)。
  - **Cost Metrics (成本度量)**：Cost Variance (成本偏差)、ROI (Return on Investment, 投资回报率)。
  - **Efficiency Metrics (效率度量)**：Productivity (生产力 - 功能点/工时)、Review Efficiency (复审效率)。
  - **Team & Customer (团队与客户)**：Team Morale (团队士气)、User Feedback (用户反馈)、NPS (Net Promoter Score, 净推荐值)。

## 六、 评估指标实例 (Evaluation Examples)

- **Traceability (可追溯性)**：文档化的金字塔结构（需求 -> 特性 -> 用例 -> 场景 -> 测试用例）。
- **High Cohesion & Low Coupling (高内聚低耦合)**：设计分析的重要指标。
- **Cyclomatic Complexity (圈复杂度)**：
  - **Formula (公式)**：$V(G) = E - N + 2P$ ($E$: 边数, $N$: 节点数, $P$: 连通分量数)。
  - **Significance (意义)**：量化独立路径数量，复杂度高意味着理解维护难、Bug 风险高、测试工作量大。
- **LOC Time-series Variation (LOC 时序变化)**：以正常状态为 Benchmark (基准)，通过偏差判断流程运行是否顺利。
- **Coverage Matrix (覆盖矩阵)**：通过 UC (用例) 与 TC (测试用例) 的映射表确保功能点全覆盖。
- **Rayleigh Distribution (瑞利分布)**：用于描述缺陷在一个周期内的时序变化趋势，预测质量状态。

---

## 一、 测试生命周期与目标 (Test Life Cycle & Objectives)

- **Test Life Cycle (测试生命周期)**:
  - **Planning (计划阶段)**: Test Objective (测试目标), Estimation (估算), Strategy (策略), Process (流程), Schedule (进度).
  - **Requirements (需求阶段)**: Test Requirements (测试需求), Testability Analysis (可测试性分析), Requirement Traceability (需求可追溯性), Change Control (变更控制), Issue Tracking (问题跟踪).
  - **Design & Development (设计与开发阶段)**: Environment Setup (环境搭建), Test Case/Data Design (用例/数据设计), Automation Design (自动化设计), Scripts Development (脚本开发).
  - **Execution & Analysis (执行与分析阶段)**: Smoke Test (冒烟测试), Test Execution (测试执行), Defect Tracking (缺陷跟踪), Test Analysis (测试分析), Test Reporting (测试报告).
- **Test Case Design Objective (测试用例设计目标)**:
  - **Good Test Coverage (良好的测试覆盖率)**: 确保测试用例 (TC) 覆盖所有功能特性和需求 (UC)。
  - **Executability and Maintainability (可执行性与可维护性)**: 组织结构和层级应合理，降低执行和维护成本。
  - **Good Test Traceability (良好的测试可追溯性)**: 遵循 Needs (需求) -> Features (特性) -> Use Cases (用例) -> Scenarios (场景) -> Test Cases (测试用例) 的金字塔结构。
  - **End User Needs (最终用户需求)**: 始终牢记用户核心需求。
- **Domain & Business Knowledge (领域与业务知识)**: 测试技术的有效性取决于测试人员对系统的理解和在正确场景下应用技术的能力。

## 二、 等价类划分法 (Equivalence Class Partitioning - ECP)

- **Theory (原理)**: 将输入域划分为若干子集，每个子集内的数据对于揭示程序错误是等效的。
  - **Completeness (完备性)**: 所有子集之和覆盖整个输入域。
  - **No Duplication (无冗余)**: 子集之间互不相交。
- **Partitioning (划分方式)**:
  - **Valid class (有效等价类)**: 符合规格说明的合理输入。
  - **Invalid class (无效等价类)**: 不符合规格说明的非法输入。
- **Partitioning Principles (划分原则)**:
  - **Range (范围)**: 如 1 到 999 之间的整数。
  - **Group (集合)**: 如车辆列表包括卡车、轿车、挂车。
  - **Specific/Unique (特定值/唯一值)**: 如必须包含 "@" 符号。
- **Special Conditions (特殊情况处理)**:
  - **Default (默认值)**: 未提供输入时假设的值。
  - **Empty (空值)**: 存在但无内容，如空字符串 `""`。
  - **Blank (空格)**: 存在且有内容，如包含空格的字符串 `" "`。
  - **Null (空引用)**: 值不存在或未分配。
  - **Zero (零)**: 数值 0。
  - **None (无选择)**: 从列表中不选任何项。
- **Approach (操作步骤)**: 1. 划分子集并编号。 2. 编写测试用例覆盖所有有效类（一个用例可覆盖多个有效类）。 3. 编写测试用例覆盖无效类（**每个用例仅覆盖一个无效类**，其余参数取有效值）。
- **Single Fault Assumption (单故障假设)**: 假设失败很少由两个及以上故障同时发生引起，因此测试时通常一次只改变一个无效变量。

## 三、 边界值分析法 (Boundary Value Analysis - BVA)

- **Rationale (理论依据)**: 经验表明错误最常发生在输入域的边界及其附近（如循环结构、关系运算符 `<` 与 `<=`）。
- **Boundary Types (边界类型)**:
  - **Baseline (基准测试)**: 关注边界上的值，包括 $min, min+, nominal (名义值), max-, max$。
  - **Robustness (健壮性测试)**: 增加超出边界的值，包括 $min-$ 和 $max+$。
  - **Formula (公式)**: 对于 $N$ 个变量，健壮边界测试的用例数为 $6N + 1$。
- **Hidden Boundary Values (隐藏边界值)**: 由于参数间的 **Dependency (依赖关系)** 导致的边界，例如 2 月没有 30 日或 31 日，闰年与平年对 2 月 29 日的影响。
- **Output Boundary (输出边界)**: 不仅考虑输入边界，还应考虑系统输出的极限值。

## 四、 组合分析/两两组合法 (Combinatorial Analysis / Pairwise)

- **Problem (组合爆炸问题)**: 当变量及取值较多时，全组合测试用例数巨大。
- **Rationale (基本原理)**: 大多数故障是由单个参数或 **Two-way interaction (两两交互)** 引起的，多参数（3路以上）触发的故障比例极低。
- **Pairwise Testing (两两组合测试)**: 确保每一对参数的所有取值组合至少被覆盖一次。
- **n-Way Testing (n路组合测试)**: 覆盖任意 $n$ 个参数的所有取值组合，随着 $n$ 增加，覆盖率提高但成本剧增。
- **Application Scenarios (应用场景)**:
  - **Configuration (配置测试)**: 不同操作系统、语言、架构的组合。
  - **API Testing (接口测试)**: 多个参数及其取值的组合。
  - **GUI (图形用户界面)**: 界面设置项（如下拉框、单选框）的相互影响。

---

## 一、 测试执行基础 (Basic Test Execution)

- **Test Execution (测试执行)**：根据被测软件处理测试用例集并产生结果的过程。
- **Execution Timing (执行时机)**：测试执行位于冒烟测试 (Smoke Test) 之后，缺陷跟踪 (Defect Tracking) 和测试分析 (Test Analysis) 之前。
- **Test Environment (测试环境)**：测试执行发生的特定地点和技术配置。
- **Suspension Criteria (暂停准则)**：导致测试活动暂时停止的条件：
  - 硬件或软件在计划时间内不可用。
  - 构建版本包含严重缺陷，严重阻碍或限制测试进度。
  - 分配的测试资源在团队需要时不可用。
- **Resumption Criteria (恢复准则)**：测试活动重新开始的标准：
  - 只有当导致暂停的问题得到解决时才恢复测试。
  - 若因关键缺陷导致暂停，在恢复测试前必须由测试团队验证“修复 (FIX)”。

------

## 二、 缺陷报告与跟踪 (Defect Reporting and Tracking)

- **Bug Reporting Principle (缺陷报告原则)**：没有任何 Bug 是微不足道而不必报告的。
- **Defect Attributes (缺陷属性)**：报告中包含的关键信息项：
  - **Summary (摘要)**：对缺陷的简洁描述，常用于搜索。
  - **Description (描述)**：包含重现步骤 (Steps to reproduce)、实际结果 (Actual Result) 和预期结果 (Expected Result)。
  - **Severity (严重程度)**：对业务影响的大小。
  - **Priority (优先级)**：修复缺陷的紧迫程度（P1 最高，P5 最低）。
  - **Status (状态)**：缺陷在生命周期中的当前位置。
- **Effective Defect Writing Principles (有效缺陷编写原则)**：
  - **Condense (精炼)**：简洁明了。
  - **Accurate (准确)**：描述客观。
  - **Neutralize (中立)**：不带个人情感。
  - **Precise (精确)**：细节详尽。
  - **Impact (影响)**：说明后果。
  - **Re-create (可重现)**：确保他人可复现。
  - **Evidence (证据)**：提供截图或日志。

------

## 三、 缺陷严重程度分级 (Defect Severity Levels)

| **级别 (Level)** | **定义 (Definition)**              | **业务关键性 (Business Critical)** | **流程停止 (Process Stop)** | **示例 (Example)**                 |
| ---------------- | ---------------------------------- | ---------------------------------- | --------------------------- | ---------------------------------- |
| **S1**           | 业务关键流程停止，且无可行替代方案 | 是 (YES)                           | 是 (YES)                    | 无法下单或过账                     |
| **S2**           | 业务关键流程出现非停止类缺陷       | 是 (YES)                           | 否 (NO)                     | 价格计算错误或手动绕过方案极其繁琐 |
| **S3**           | 非业务关键流程停止，无可行替代方案 | 否 (NO)                            | 是 (YES)                    | 无法复制订单或告警未发布           |
| **S4**           | 非业务关键流程出现非停止类缺陷     | 否 (NO)                            | 否 (NO)                     | 查询时用户被随机踢出系统           |
| **S5**           | 不影响正常业务运行的次要缺陷       | 否 (NO)                            | 否 (NO)                     | 无意义的错误消息                   |
| **S6**           | 增强建议 (Enhancement Request)     | N/A                                | N/A                         | 在报告中添加新列                   |

------

## 四、 缺陷生命周期管理 (Bug Life Cycle Management)

- **Key Statuses (关键状态)**：
  - **Unconfirmed (未确认)**：新 Bug 需经 QA 验证其是否存在。
  - **New (新建)**：QA 确认 Bug 存在。
  - **Assigned (已分配)**：开发人员认领并负责处理。
  - **Resolved (已解决)**：修复完成，等待验证。
  - **Verified (已验证)**：QA 确认修复奏效。
  - **Reopen (重开)**：验证发现缺陷未修复。
  - **Closed (关闭)**：产品发布后彻底关闭。
- **Possible Resolutions (可能的解决方案)**：
  - **FIXED**：已修复。
  - **DUPLICATE**：重复缺陷。
  - **WONTFIX**：不予修复。
  - **WORKSFORME**：无法重现。
  - **INVALID**：无效缺陷。

------

## 五、 矩阵与参与者 (Matrix & Participants)

- **Defect Tracking Matrix (缺陷跟踪矩阵)**：规定了不同状态下的操作者 (Owner) 和参与者。
- **Participants (参与者职责)**：
  - **QA (测试人员)**：负责报告 (New)、复现失败确认 (Cannot Duplicate)、验证修复 (Verified/Closed) 和重开 (Reopen)。
  - **DEV (开发人员)**：负责认领 (Assigned)、修复 (Fixed) 或标识为重复/非 Bug。
  - **PM (项目经理)**：负责决定缺陷是否延期处理 (Deferred)。
  - **BA (业务分析师)**：协助澄清需求，判断是否为 Bug。/

------

## 一、 测试提交概述 (Test Submission Overview)

- **Potentially Shippable Product Increment (潜在可交付产品增量)**：Sprint 结束时的整体提交物。
- **Runnable Software Package (可运行软件包)**：包括 Application Code (应用代码)、Operations Code (运维代码) 和 Test Code (测试代码)。
- **Complete Documentation (整套文档)**：
  - **Requirements Documentation (需求文档)**：当前 Sprint 执行的 Backlog 及所有历史记录。
  - **Development Documentation (开发文档)**：包含技术路线、思路及整体架构。
  - **Test Documentation (测试文档)**：测试过程、结果以及所有的 Regression (回归测试)。
  - **Operations Documentation (运维文档)**：代码包的完整执行指南。
- **QA Deliverables (QA 提交物清单)**：需求对应文档、Test Case Suite (测试用例集)、Test Case Inventory (用例清单)、带日志的用例文档、Bug/Issue List (缺陷/问题列表)、Test Report (测试报告)。
- **Configuration Management (配置管理)**：所有提交物需纳入配置管理，确保按版本时序排列并带有清晰标签。

## 二、 质量报告要素 (Elements of Quality Report)

- **Three-tier Reporting System (三级报告体系)**：涵盖代码、缺陷 (Bug)、性能、流程、团队等维度的深度分析。
- **Elements of a Good Report (优秀报告的要素)**：
  - **Subject (主题)**：明确主题的定义、范围和组成部分。
  - **Audience (对象)**：考虑报告对象的职级、背景及其组合。
  - **Viewpoint (观点)**：观点应具备导向性、主观性与伸缩性。
- **Data Analysis Dimensions (数据分析维度)**：总量 (Total)、等级 (Grade)、时序变化 (Time-series Variation，如日/年/版本)、分布 (Distribution)、对比 (Comparison)。

## 三、 软件度量指标 (Software Metrics & Indicators)

- **Metric Objectives (度量目标)**：改进质量、控制成本、提高效率。
- **Product Metrics (产品度量)**：
  - **Size (规模)**：LOC (Lines of Code, 代码行数)、FP (Function Points, 功能点)。
  - **Quality (质量)**：Defect Density (缺陷密度)、Reliability (可靠性，如 MTTF)、Maintainability (维护性)、Testability (可测试性，如自动化覆盖率)。
- **Process Metrics (过程度量)**：
  - **Progress (进度)**：Schedule Variance (计划偏差)、Critical Path Completion Rate (关键路径完成率)。
  - **Cost (成本)**：Cost Variance (成本偏差)、ROI (投资回报率)。
  - **Efficiency (效率)**：Productivity (生产力)、Review Efficiency (复审效率)。
- **Quality Logic (质量判断逻辑)**：
  - **Defects vs. Quality (缺陷与质量)**：缺陷越严重/越多，说明质量越差，且预示着潜伏的未知缺陷更多。
  - **Complexity vs. Quality (复杂度与质量)**：系统越复杂（功能、设计、代码），质量通常越差。
  - **Process vs. Quality (流程与质量)**：与计划偏差越大、自动化/标准化程度越低，质量越差。

## 四、 发布评审与决策 (Release Review & Decision)

- **Release Review Meeting (发布评审会议)**：基于客观数据讨论并做出决策。
- **Quality Analysis (质量分析判断依据)**：
  - **Stability (稳定性)**：质量随迭代稳步提升，并进入 Plateau Phase (平台期)。
  - **Severity (严重性)**：高等级缺陷（如 1、2 级）已被修复，低级别缺陷占比极高（如超过 99%）。
  - **Trend (趋势)**：总缺陷数趋于稳定，后期迭代增加量极低（如低于 1%）。
  - **Residual Risk (残留风险)**：未修复缺陷全部集中于 Minor (次要) 级别以下，且占总量比例极低（如低于 0.4%）。
- **Decision (决策建议)**：若满足上述指标，建议 Release to Production (发布到生产环境)。

------

## 一、 测试自动化定义 (Test Automation Definition)

- **Test Automation (测试自动化)**：使用软件来支持测试管理、控制测试执行以及辅助测试分析。
- **Management (管理)**：涉及 Planning (计划) 与 Evaluation (评估)。
- **Implementation (执行)**：涉及 Design (设计) 与 Execution (执行)。
- **Analysis (分析)**：涉及 Reporting (报告) 与 Analysis (分析)。
- **Core Characteristics (领域特点)**：包括 ROI (投资回报率) 估算、数据驱动与关键字驱动、架构设计以及项目管理。

## 二、 自动化准入与 ROI (ROI & Admission Criteria)

- **ROI Model (投资回报率模型)**：
  - **Waterfall conditions (瀑布条件下)**：假设开发模型顺利且自动化能完全代替手工 GUI 测试时，面积表示努力程度 (Effort)。
  - **Iterative conditions (迭代条件下)**：强调在回归测试 (Regression) 中节省的时间，自动化能显著降低重复劳动的成本。
- **Effort for Automation (自动化努力方向)**：尽可能的提前准入点 (Admission Point)、缩短设计开发时间、并尽可能多地执行。
- **When to Automate (适用场景)**：
  - 处理耗时、重复、无趣的工作。
  - 执行手工难以完成的任务，如性能测试、安全性测试、高可用性测试、安装测试及界面对比。
- **Empirical Admission Criteria (实践性准入条件)**：
  - **Stable System Under Test (相对稳定的被测系统)**。
  - **Well-designed Test Case Inventory (设计良好的测试用例清单)**。

## 三、 脚本编写与框架设计 (Scripting & Frameworks)

- **Basic Scripts (基础脚本模式)**：
  - **Recording/Playback (录制与回放)**：录制用户的操作序列并回放。
  - **Linear Script (线性脚本)**：顺序执行的操作指令，缺乏灵活性。
- **Structured Scripting (结构化脚本)**：使用函数 (Functions) 或子程序 (Subroutines) 封装常用操作，增加复用性。
- **Data-Driven Testing (数据驱动测试)**：
  - **Definition (定义)**：测试输入和预期结果存储在外部数据文件（如 Excel、CSV）中。
  - **Mechanism (机制)**：脚本通过循环读取外部数据来驱动测试过程，实现同一逻辑多组数据的测试。
- **Keyword-Driven Testing (关键字驱动测试)**：
  - **Definition (定义)**：将测试逻辑描述为一系列业务关键字（如 "Login", "Click", "Verify"）。
  - **Benefits (优点)**：将测试逻辑与实现细节分离，使非技术背景人员也能通过组合关键字编写自动化脚本。

## 四、 自动化测试工具分类 (Automation Tools Classification)

- **Test Management (测试管理)**：Rational Test Manager, Mercury Quality Center (QC)。
- **Defect Tracking (缺陷跟踪)**：Bugzilla, JIRA, ClearQuest。
- **Unit Test (单元测试)**：JUnit, TestNG。
- **Functional Test (功能测试)**：Selenium, Rational Functional Tester (RFT), Mercury Quick Test Professional (QTP)。
- **Performance Test (性能测试)**：JMeter, LoadRunner, Grinder。
- **Rational Toolset (Rational 家族工具集)**：Functional Tester, Robot Tester, Purify, PureCoverage。

------

## 一、 非确定性结果系统 (Non-deterministic Outcome Systems)

- **Definition (定义)**：相同输入下可能产生不同输出结果的系统。
- **Influencing Factors (影响因素)**：User Intent (用户自身意愿)、Generative Models (生成式模型)、User Expression/Input (用户表达/输入)、System Output Framework (系统输出框架)。
- **Classification (分类)**：
  - **Concurrent & Multi-threaded Systems (并发与多线程系统)**。
  - **AI & Machine Learning Systems (人工智能与机器学习系统)**。
  - **Rule & Real-time Systems (规则与实时系统)**：如推荐系统、搜索系统、IoT 系统。
  - **Blockchain Consensus Systems (区块链共识系统)**。

## 二、 非确定性测试目标与挑战 (Testing Objectives & Challenges)

- **Testing Objectives (测试目标)**：
  - **Verify Functional Correctness (验证功能正确性)**：输出符合业务需求。
  - **Ensure Robustness (保障鲁棒性)**：抵御扰动与对抗样本。
  - **Ensure Fairness (确保公平性)**：消除偏见（如性别、种族）。
  - **Improve Security (提升安全性)**：防范窃取、泄露与攻击。
  - **Verify Generalization (验证泛化能力)**：在未见数据上表现稳定。
  - **Control Non-determinism (控制非确定性)**：波动在可接受范围内。
- **Core Challenges (核心难题)**：
  - **Non-determinism (非确定性)**：随机初始化或数据增强导致输出不一。
  - **Black-box Characteristics (黑盒特性)**：决策逻辑难以解释。
  - **Data Dependency (数据依赖性)**：高度依赖训练/测试数据质量。
  - **Dynamic Nature (动态性)**：数据漂移导致性能退化。

## 三、 最佳实践与处理方法 (Best Practices & Methods)

- **Handling Non-determinism (处理非确定性)**：
  - **Fix Random Seeds (固定随机种子)**。
  - **Statistical Fluctuation Range (多次运行统计波动范围)**。
  - **Set Error Thresholds (设置合理容错阈值)**。
- **Result Consistency Testing (结果一致性测试)**：
  - 多次运行相同输入（如 100 次），统计波动。
  - 使用自定义断言脚本，如对比文本的 Cosine Similarity (余弦相似度)。

## 四、 模型训练与推理参数 (Model Training & Inference Parameters)

- **Model Structure Parameters (模型结构参数)**：Hidden Dimensions (隐藏层维度)、Attention Heads (注意力头数)、Network Layers (网络层数)。
- **Optimizer Parameters (优化器参数)**：Optimizer Type (优化器类型)、Weight Decay (权重衰减)、Learning Rate (学习率)、Gradient Clipping (梯度裁剪)。
- **Data & Regularization (数据与正则化)**：Dropout Probability (Dropout 概率)、Sampling Strategy (数据采样策略)、Sequence Length (序列长度截断/填充)。
- **Inference Adjustment (推理修正/模型应用)**：Temperature (温度系数)、Prompt Engineering (提示词工程)、Templates (模版)。

## 五、 各类功能性测试 (Functional Testing Categories)

- **Data Testing (数据测试)**：AI 测试的基础，验证模型“原料”。
- **Model Training Testing (模型训练测试)**：验证训练过程及改进的有效性。
- **Model Inference Testing (模型推理能力测试)**：验证推理输出质量。
- **Scenario Coverage Testing (场景覆盖测试)**：覆盖正常、边界及异常业务场景。
- **Model Performance Testing (模型性能测试)**：验证响应速度与资源占用。
- **Model Security Testing (模型安全性测试)**：防御恶意攻击与数据安全。

---

## 性能工程与测试 (Performance Engineering & Testing)

### 1. 核心概念与基本问题 (Core Concepts & Fundamental Issues)

- **Performance Engineering**: 性能工程。包含容量规划 (Capacity Planning)、架构设计 (Architecture Design)、性能测试 (Performance Testing) 与性能优化 (Performance Optimization)。
- **Fundamental Issues of Performance**: 性能的基本问题。主要关注速度快慢 (Speed)、规模大小 (Scale/Magnitude) 以及运行稳定性 (Stability)。
- **Balance between Business and Performance**: 业务与性能的平衡。需考虑正确性 (Correctness) 和功能性 (Functionality)。
- **High-level Design Patterns (Architecture)**: 高等级设计模式（架构）。涉及分层 (Layering)、分布式 (Distributed)、通信模式 (Communication Patterns)、多级缓存 (Multi-level Caching)、多级持久化 (Multi-level Persistence) 及线上线下处理。
- **Low-level Design Patterns (GoF23)**: 低等级设计模式。如工厂模式 (Factory)、单例模式 (Singleton)、锁机制（特别是排他锁 Exclusive Lock）及动态/静态处理。
- **Hardware Resource Planning**: 硬件资源规划。包括 CPU、内存 (Memory)、磁盘转速 (Disk Speed)、带宽 (Bandwidth) 及云计算 (Cloud Computing)。

------

### 2. 性能测试指标 (Performance Metrics)

- **Concurrency**: 并发量 (Q)。
- **QPS (Query Per Second)**: 每秒查询数。指一个请求、一个查询。
- **RrPS (Request Per Second)**: 每秒请求数。
- **Accuracy / Success Rate**: 正确率。
- **Throughput (T)**: 吞吐量。涉及前端与后端，需判断 T 是否为 Q 的组合。
- **TPS (Transactions Per Second)**: 每秒事务数。需关注页面与事务之间的关系。
- **Response Time (RT)**: 响应时间。由处理时间 (Processing Time)、传输时间 (Transfer Time) 和绘制时间 (Rendering/Drawing Time) 组成。
  - **ART**: 平均响应时间 (Average Response Time)。
  - **AART**: 绝对平均响应时间 (Absolute Average Response Time)。

------

### 3. 测试设计与系统评价 (Test Design & Evaluation)

- **Test Design**: 测试设计。包含用户行为建模（如登录、搜索、加购、支付等流程的转化率与路径）。
- **System Performance Evaluation**: 系统性能评价。
  - **Single test vs. Incremental test**: 一次性测试还是随时间增加。
  - **System Initialization**: 系统是否初始化及如何初始化。
  - **Concurrent Users**: 并发用户数与 RrPS 的关系。
- **Load Analysis Concepts**:
  - **Light Load**: 轻负载。
  - **Heavy Load**: 重负载。
  - **Buckle Zone**: 崩溃区。
  - **The Optimum Number of Concurrent Users**: 最佳并发用户数。
  - **The Maximum Number of Concurrent Users**: 最大并发用户数（此时资源饱和 Resource Saturated、吞吐量下降 Throughput Falling、终端用户受影响 End Users Effected）。

------

### 4. 关键考虑因素 (Other Key Considerations)

- **Session Persistence**: 持久化 Session、保持登录、验证码处理。
- **Caching**: 是否使用缓存。
- **Think Time**: 思考时间与业务需求。
- **Combination**: 业务逻辑的组合与再组合。

------

### 5. 性能测试工具 (Performance Testing Tools)

- **Comprehensive Performance Testing Tools**: 综合性能测试工具（负载/压力测试）。如 JMeter, LoadRunner, Gatling, Locust。
- **Lightweight Interface/Throughput Tools**: 轻量型接口/吞吐量测试工具。如 Apache Bench (ab), Postman, WRK。
- **Performance Monitoring & Analysis**: 性能监控与分析工具。如 Prometheus+Grafana, New Relic, Chrome DevTools。
- **Specialized Performance Tools**: 专项性能测试工具。如 Redis-benchmark, MySQL Slap。

---

好的，完全明白。我会严格遵守你的要求：**不标注任何来源**，保持内容**紧凑**，并采用**英文优先、中文对照**的格式。

以下是针对《安全测试》PPT整理的考点笔记：

## Security Testing (安全测试)

### 1. Fundamental Definitions (基本定义)

- **Security Testing**: 安全测试。发现软件系统及数据相关的安全漏洞 (**Security Vulnerabilities/Bugs**) 和安全隐患 (**Security Issues**)。
- **Fragility and Uncontrollability**: 脆弱与不可控。由于开发人员错误导致系统在面对攻击时表现出的状态。
- **Foundation**: 基础。安全测试建立在安全管理 (**Security Management**) 和安全审计 (**Security Audit**) 之上。

### 2. Responsibility and Scope (职责与范围)

- **R&D Team (Testing Scope)**: 研发团队（测试范围）。关注系统 (**System**) 和数据 (**Data**)。
- **Architect/OP (Audit Scope)**: 架构师/运维（审计范围）。关注中间件 (**Middleware**)、运行时 (**Runtime**) 和操作系统 (**O/S**)。
- **Infra (Management Scope)**: 基础设施（管理范围）。关注节点 (**Nodes**) 和网络 (**Networking**)。
- **Service Models**: 云服务模式下的管理权限对比。
  - **On-Premises**: 本地部署。
  - **IaaS (Infrastructure as a Service)**: 基础设施即服务。
  - **PaaS (Platform as a Service)**: 平台即服务。
  - **SaaS (Software as a Service)**: 软件即服务。

------

### 3. Four Categories of Security (四大安全性分类)

#### (1) Execution-related Security (与执行相关的安全性)

- **Principle**: 利用输入漏洞注入可执行代码。
- **SQL Injection**: SQL 注入。操纵 SQL 逻辑，导致登录绕过或数据泄露（如使用 `' OR 'x'='x'`）。
- **XSS (Cross-Site Scripting)**: 跨站脚本攻击。
- **File Upload Vulnerability**: 文件上传漏洞。
- **Prompt Injection**: 大模型提示词注入漏洞。
- **Tools**: **SQLiDetector** (模式检测)、**SQLMap** (基于风险的自动化检测)、**OWASP ZAP** (代码扫描)。

#### (2) Permission-related Security (与权限相关的安全性)

- **Principle**: 伪造或劫持 Session/Cookie 窃取权限。
- **Session-Cookie Spoofing**: Session-Cookie 诱骗。核心防护：访问控制、传输加密、凭证安全存储。
- **XSS Theft**: 通过脚本读取 `document.cookie` 并发送给攻击者服务器。
- **OAuth Attack**: 第三方认证攻击。
- **Testing Methods**: 验证 **HttpOnly, Secure, SameSite** 属性，检查传输层安全。

#### (3) Stress-related Security (与压力相关的安全性)

- **Principle**: 高频并发抢占系统资源（带宽、CPU、内存、FD）。
- **DDoS (Distributed Denial of Service)**: 分布式拒绝服务。特征：分布式、资源耗尽。
- **CC Attack (Challenge Collapsar)**: 饱和攻击。
- **Resource Limits**: 资源上限。
  - **File Descriptor (FD)**: 文件描述符。Linux 默认单进程限制（通常为 1024）。
  - **TCP Quaternary**: TCP 四元组（源IP、源端口、目的IP、目的端口）。

#### (4) Information-related Security (与信息相关的安全性)

- **Principle**: 利用信息不对称非法获取资源。
- **Placeholder Attack**: 占位攻击。包括身份占位和资源占位。
- **Malicious Crawler**: 恶意爬虫。
- **Cache Poisoning**: 缓存污染。

------

### 4. Security Audit (安全审计)

#### (1) Configuration Audit (配置项审计)

- **Audit Items**: 操作系统、Web服务器、数据库、FTP/SSH工具。
- **Key Risks**:
  - **Plaintext Credentials**: 明码用户名密码。
  - **Built-in IDs**: 内建 ID。
  - **Absolute Path Storage**: 使用绝对地址。
  - **Inconsistency**: 配置项不一致。

#### (2) Operations & Dependency Audit (运维与依赖项审计)

- **Operations Risks**: 暴露运维入口至 DMZ 外、未经修改的 admin 密码、开发后门。
- **Dependency Risks (Platform)**: 使用有漏洞的依赖项。
  - **Heartbleed (OpenSSL 1.0)**: 心脏滴血漏洞。
  - **Struts2 RCE**: 远程执行漏洞。
  - **SSRF (Ueditor)**: 服务端请求伪造。

---

这份关于 **单元测试 (Unit Test)** 的期末复习笔记已重新整理。去除了所有来源标注，采用最紧凑的格式，并确保英文术语在前，中文解释在后，方便您考试时直接引用。

------

## 1. 单元测试定义 (Definition of Unit Test)

- **Unit Testing**: A software testing method by which individual units of source code are tested to determine whether they are fit for use. (单元测试是一种测试方法，通过对源代码中的独立单元进行测试，以确定它们是否可以投入使用。)
- **Components**: Sets of one or more computer program modules together with associated control data, usage procedures, and operating procedures. (由一个或多个程序模块、相关的控制数据、使用程序和操作程序组成。)
- **Isolation**: To isolate each part of the program and show that the individual parts are correct. (隔离程序的每个部分，并证明这些独立部分是正确的。)
- **Contract**: Provides a strict, written contract that the piece of code must satisfy. (提供代码必须满足的严格的书面契约。)

------

## 2. 核心价值 (Why We Need Unit Test)

- **Bring the test as early as possible**: Kill defects before they cost you; the relative cost of fixing defects increases significantly in later stages (e.g., In service). (尽早测试：在缺陷产生高昂代价前将其消除；后期修复成本远高于早期。)
- **Make the white box test possible**: Allows for testing internal logic at the most granular level. (使白盒测试成为可能：允许在最细粒度层面测试内部逻辑。)
- **Lower the threshold of test**: Helps manage constraints like Time, Source, Environment, Resource, and Scope. (尽可能降低测试门槛：平衡时间、源码、环境、资源和范围等因素。)
- **Make the integration as specific as possible**: Ensures that individual parts are functional before they are combined. (使集成更具体：确保在集成前各独立部分功能正常。

------

## 3. 开发模式对比 (Development Patterns)

| **Characteristic (特征)** | **General Development (常规开发)** | **TDD (测试驱动开发)**            | **XP (极限编程)**               |
| ------------------------- | ---------------------------------- | --------------------------------- | ------------------------------- |
| **Timing (时机)**         | After code developed (开发后)      | Before code written (编写前)      | Continuously (持续进行)         |
| **Coverage (覆盖率)**     | Defined by developer, low (较低)   | Typically high (通常较高)         | High, automated (高且自动化)    |
| **Feedback (反馈)**       | Slower (较慢)                      | Immediate (即时)                  | Fast (快速)                     |
| **Design (设计)**         | Less refactoring (较少重构)        | Tests drive design (测试驱动设计) | Frequent refactoring (频繁重构) |
| **Ownership (所有权)**    | Independent (独立)                 | Guides design (引导设计)          | Collective ownership (集体所有) |

------

## 4. 执行步骤 (How to Perform Unit Test)

1. **Frameworks**: Use a unit test framework and related libraries. (使用单元测试框架及相关库。)
2. **Mocking**: Simulate the surrounding environment by using mock. (使用 Mock 技术模拟周边环境。)
3. **Scripts**: Develop a unit test case script for every unit with independent tests and mark these tests. (为每个单元开发独立的测试脚本并标记。)
4. **Assertion**: Assert the output of the tests (e.g., `assertEquals`). (对测试输出进行断言，验证结果是否符合预期。)
5. **Maintenance**: Updating based on change and running regularly. (根据变更持续更新并定期运行。)

------

## 5. 单元测试与白盒测试的区别 (Unit Test vs. White Box Test)

- **Scope**: Unit test focus on the unit as a whole, while white box test focus on the logic. (单元测试关注单元整体，白盒测试关注内部逻辑。)
- **Method**: Unit test is dynamic; white box includes both static and dynamic methods. (单元测试是动态的；白盒测试包含静态和动态方法。)
- **Nature**: Unit test is a process/phase; white box test is a method. (单元测试是一个过程/阶段；白盒测试是一种方法。)
- **Language**: Unit test is programming language sensitive; white box test is not. (单元测试对编程语言敏感；白盒测试则不敏感。)
- **Focus**: Unit test focus on input and output; white box test focus on logic paths. (单元测试关注输入输出；白盒测试关注逻辑路径。)