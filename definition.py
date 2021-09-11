import enum


class Actions(enum.Enum):
    Wait = "等待"  # 正在执行操作, 不需要执行任何操作
    Touch = "点击"  # 正在执行操作, 需要点击屏幕
    Finish = "完成"  # 当前操作已完成, 进行下一个任务
    BackHome = "返回主页"  # 当前操作失败, 下一步会自动跳回主页, 比如检查邮件遇到登陆弹窗
    # Skip = "跳过"  # 跳过当前操作, 比如说理智不足无法战斗
    # NoRational = "理智不足"


class Tasks(enum.Enum):
    Home = "主页"
    Email = "邮件"
    Annihilation = "剿灭"
    RecentBattle = "最近关卡"
    Battle = "战斗"
    Foundation = "基建"
    Task = "任务"
    FriendFoundation = "好友基建"
    Shopping = "购物"


ACTIONS = sorted([i.value for i in Actions])
TASKS = sorted([i.value for i in Tasks])
