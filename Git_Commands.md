# Git 命令参考指南

本文档包含与 GitHub 互动的常用 Git 命令，用于海岛能源优化项目的版本控制。

## 基础配置

```bash
# 配置用户信息（首次使用）
git config --global user.name "Le vent"
git config --global user.email "1813088239@qq.com"

# 查看配置
git config --list
```

## 仓库初始化与克隆

```bash
# 初始化本地仓库
git init

# 克隆远程仓库
git clone https://github.com/username/repository-name.git

# 添加远程仓库
git remote add origin https://github.com/username/repository-name.git

# 查看远程仓库
git remote -v
```

## 文件操作

```bash
# 查看文件状态
git status

# 添加文件到暂存区
git add filename.txt          # 添加单个文件
git add .                     # 添加所有文件
git add *.py                  # 添加所有 Python 文件

# 取消暂存
git reset filename.txt        # 取消暂存单个文件
git reset                     # 取消暂存所有文件
```

## 提交操作

```bash
# 提交更改
git commit -m "Add energy optimization model"

# 添加并提交（跳过暂存区）
git commit -am "Update disaster modeling code"

# 修改最后一次提交信息
git commit --amend -m "New commit message"
```

## 分支管理

```bash
# 查看分支
git branch                    # 查看本地分支
git branch -r                 # 查看远程分支
git branch -a                 # 查看所有分支

# 创建分支
git branch feature-branch     # 创建新分支
git checkout -b feature-branch # 创建并切换到新分支

# 切换分支
git checkout main
git checkout feature-branch

# 合并分支
git checkout main
git merge feature-branch

# 删除分支
git branch -d feature-branch  # 删除本地分支
git push origin --delete feature-branch # 删除远程分支
```

## 远程操作

```bash
# 推送到远程仓库
git push origin main          # 推送到主分支
git push origin feature-branch # 推送到特定分支
git push -u origin main       # 首次推送并设置上游分支

# 从远程仓库拉取
git pull origin main          # 拉取并合并
git fetch origin              # 只拉取，不合并

# 查看远程分支信息
git remote show origin
```

## 查看历史

```bash
# 查看提交历史
git log                       # 详细历史
git log --oneline             # 简洁历史
git log --graph --oneline     # 图形化显示

# 查看文件更改历史
git log filename.py

# 查看具体提交的更改
git show commit-hash
```

## 差异比较

```bash
# 查看工作区与暂存区差异
git diff

# 查看暂存区与最后提交的差异
git diff --cached

# 比较两个提交
git diff commit1 commit2

# 查看特定文件的差异
git diff filename.py
```

## 撤销操作

```bash
# 撤销工作区的更改
git checkout -- filename.py

# 撤销提交（保留更改）
git reset --soft HEAD~1

# 撤销提交（丢弃更改）
git reset --hard HEAD~1

# 撤销特定提交
git revert commit-hash
```

## 标签管理

```bash
# 创建标签
git tag v1.0                  # 轻量标签
git tag -a v1.0 -m "Version 1.0 release" # 注释标签

# 查看标签
git tag

# 推送标签
git push origin v1.0          # 推送特定标签
git push origin --tags        # 推送所有标签

# 删除标签
git tag -d v1.0               # 删除本地标签
git push origin --delete tag v1.0 # 删除远程标签
```

## GitHub 特定操作

```bash
# 创建 Pull Request（通过 GitHub CLI）
gh pr create --title "Add new feature" --body "Description of changes"

# 查看 Pull Requests
gh pr list

# 合并 Pull Request
gh pr merge PR-number

# 创建 Issue
gh issue create --title "Bug report" --body "Issue description"

# 查看 Issues
gh issue list
```

## 项目特定工作流

```bash
# 开发新功能的典型流程
git checkout main
git pull origin main
git checkout -b feature/optimization-improvement
# ... 进行开发工作 ...
git add .
git commit -m "Improve energy optimization algorithm"
git push -u origin feature/optimization-improvement
# ... 在 GitHub 上创建 Pull Request ...

# 发布版本的典型流程
git checkout main
git pull origin main
git tag -a v2.0 -m "Version 2.0: Add disaster modeling"
git push origin v2.0
git push origin main
```

## 常用快捷命令组合

```bash
# 快速提交所有更改
git add . && git commit -m "Update analysis notebooks"

# 同步远程仓库
git fetch origin && git merge origin/main

# 创建并推送新分支
git checkout -b new-feature && git push -u origin new-feature
```

## 文件忽略

在根目录创建 `.gitignore` 文件：

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv/

# Jupyter Notebook
.ipynb_checkpoints

# Data files
*.csv
*.nc
*.xlsx

# Output files
output_*/
result/

# IDE
.vscode/
.idea/
```

## 紧急情况处理

```bash
# 临时保存工作进度
git stash
git stash pop                 # 恢复保存的进度

# 强制覆盖本地更改
git fetch origin
git reset --hard origin/main

# 查看文件的提交作者
git blame filename.py
```

---

**注意事项：**
- 在推送前始终检查 `git status`
- 提交信息要清晰描述更改内容
- 定期与远程仓库同步避免冲突
- 重要功能开发时使用分支进行隔离