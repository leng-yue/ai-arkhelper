import enum


class Results(enum.Enum):
    Next = "next"
    Failed = "failed"
    Wait = "wait"


class Actions(enum.Enum):
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
