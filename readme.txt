这是第一次用于本地仓库和远程仓库的同步学习.
该仓库主要是一些cifar10的学习的模型,里面有10个类别,每个类别大约有5000千张带标签的图片.
直接通过echo '#cifar10 >> README.md'创建一个初始的文件
在通过git add . #来讲该本地工作区间的文件添加到临时区
再通过git commit -m'注释' #将临时区的文件更新到仓库中,
最后同 git remote add orign hettps://github.com/***建立本地仓库到远程仓库的连接
最后git push orign master,将本体仓库推送到远程仓库
