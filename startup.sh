mode="train"
if [ -n "$1" ]; then #判断第一个输入是否为空
   if [ "$1" = "train" ]; then
      mode="train"
   elif [ "$1" = "test" ]; then
        mode="test"
   fi
fi
python main_newsClassify.py --mode $mode
