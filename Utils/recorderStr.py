def recorderStr(resultPath, his, modelName, accuracy, timeCost):
    # resultPath = './Florence3DActions_data/30/gen_data.txt'
    # resultPath = './UTKinectAction/60frame/Enhance contrast/gen_data.txt'
    resultPath = resultPath
    with open(resultPath, 'a') as f:
        f.write("%s:\n" % modelName)
        f.write("accuracy:%s\n" % (his.history['accuracy']))
        f.write("loss:%s\n" % (his.history['loss']))
        f.write("TestAcc:%s\n" % accuracy)
        f.write("timeCost:%s\n" % timeCost)