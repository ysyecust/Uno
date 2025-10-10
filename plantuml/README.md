# UNO PlantUML 架构图表

本目录包含 UNO 优化框架的各种架构图表，使用 PlantUML 格式编写。

## 图表列表

### 1. 类图 (uno_class_diagram.puml)
展示 UNO 框架的主要类结构和继承关系：
- 优化算法类层次
- 模型表示类
- 线性求解器层次
- 全局化机制类

### 2. 序列图 (uno_sequence_diagram.puml)
描述完整的优化求解流程：
- 从用户输入到结果输出
- 各组件间的交互时序
- 子问题求解的详细步骤

### 3. 组件图 (uno_component_diagram.puml)
展示系统的模块化架构：
- 各层次的组件划分
- 组件间的依赖关系
- 接口和端口定义

### 4. 活动图 (uno_activity_diagram.puml)
描述优化算法的工作流程：
- 预处理阶段
- 主优化循环
- 子问题求解流程
- 收敛检查逻辑

### 5. 状态图 (uno_state_diagram.puml)
展示优化过程的状态转换：
- 初始化状态
- 优化阶段状态
- 可行性恢复阶段
- 收敛检查状态

### 6. 用例图 (uno_usecase_diagram.puml)
描述系统的功能和用户交互：
- 不同用户角色
- 系统主要功能
- 功能间的包含和扩展关系

## 如何生成图表

### 方法一：使用在线工具
1. 访问 [PlantUML 在线编辑器](http://www.plantuml.com/plantuml/uml/)
2. 复制 `.puml` 文件内容
3. 粘贴到编辑器中
4. 自动生成图表

### 方法二：使用 VS Code 插件
1. 安装 PlantUML 插件
   ```
   ext install plantuml
   ```
2. 打开 `.puml` 文件
3. 按 `Alt+D` 预览图表
4. 右键导出为 PNG/SVG

### 方法三：使用命令行工具
1. 安装 PlantUML
   ```bash
   # macOS
   brew install plantuml

   # 或使用 jar 文件
   java -jar plantuml.jar
   ```

2. 生成图表
   ```bash
   # 生成 PNG
   plantuml -tpng *.puml

   # 生成 SVG
   plantuml -tsvg *.puml

   # 生成所有格式
   plantuml -tall *.puml
   ```

### 方法四：使用 Docker
```bash
# 拉取 PlantUML 镜像
docker pull plantuml/plantuml

# 生成图表
docker run -v $(pwd):/data plantuml/plantuml *.puml
```

## 图表嵌入到文档

### Markdown 中使用
```markdown
![UNO 类图](uno_class_diagram.png)
```

### LaTeX 中使用
```latex
\includegraphics[width=\textwidth]{uno_class_diagram.pdf}
```

### HTML 中使用
```html
<img src="uno_class_diagram.svg" alt="UNO Class Diagram" />
```

## 自定义和扩展

### 修改主题
在每个 `.puml` 文件开头可以修改主题：
```plantuml
!theme cerulean
' 可选主题: plain, cerulean, materia, sketchy, cyborg, etc.
```

### 调整样式
```plantuml
skinparam backgroundColor #FEFEFE
skinparam classBackgroundColor #F0F0F0
skinparam classBorderColor #333333
```

### 添加注释
```plantuml
note right of ClassName
  这是一个注释
end note
```

## 图表维护指南

1. **保持一致性**：所有图表使用相同的主题和样式
2. **及时更新**：代码改动后同步更新相关图表
3. **模块化**：每个图表关注特定方面，避免过于复杂
4. **添加说明**：在图表中使用 note 添加必要的解释

## 相关文档

- [UNO 架构文档](../UNO_ARCHITECTURE.md) - 详细的系统架构说明
- [线性求解器集成总结](../LINEAR_SOLVERS_INTEGRATION_SUMMARY.md) - 求解器对比分析
- [PlantUML 官方文档](https://plantuml.com/) - PlantUML 语法参考