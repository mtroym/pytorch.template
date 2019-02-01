import torch

from models.backbones.Xception import Xception


def createModel(opt):
    # model = Net(opt)
    # if opt.GPU:
    #     model = model.cuda()
    # return model
    pass

refers_to  = "https://github.com/bonlime/keras-deeplab-v3-plus/blob/master/model.py"

if __name__ == '__main__':
    # a = Xception()
    # x = torch.randn((3, 3, 299, 299))
    # print(a(x).shape)
    # print(299.0 / 2 / 2 / 2 / 2)

    # sql_update_area=\
    area_now = 'NOW'
    frame_area = 1
    user_id = 100
    a = "UPDATE table_workertoday \n" + \
        "SET stay_time=2, area_now={} ,trace={}, frame_area={}, has_update=1\n".format(
            area_now, area_now, frame_area) + \
        "WHERE user_id={}".format(user_id)


    print(a)
# conn.execute(sql_update_area %(area_now ,area_now ,mj ,user_id))