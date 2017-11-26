from WindPy import *
import datetime,os


def __targ2str(arg):
    if (arg == None): return [""];
    if (arg == ""): return [""];
    if (isinstance(arg, str)): return [arg];
    if (isinstance(arg, list)): return [str(x) for x in arg];
    if (isinstance(arg, tuple)): return [str(x) for x in arg];
    if (isinstance(arg, float) or isinstance(arg, int)): return [str(arg)];
    if (str(type(arg)) == "<type 'unicode'>"): return [arg];
    return None


def mwsi(codes, fields, beginTime=datetime.datetime.now() - datetime.timedelta(days=30),
         endTime=datetime.datetime.now()):
    if (isinstance(codes, str)):
        codes = codes.split(",")
        if (codes.__len__() != 1):
            pass  # messagebox
            return
    if (isinstance(fields, str)):
        fields = fields.split(",")
    try:
        while (1):
            data = w.wsi(codes, fields, beginTime, endTime)
            if (data.ErrorCode != 0):
                pass  # messagebox
            else:
                return data

    except:
        print(u"未知网络错误")
        pass  # messagebox
        os._exit(0)


w.start()
data = mwsi("600000.SH", "close,amt", datetime.datetime.now() - datetime.timedelta(days=3))
print(data)
