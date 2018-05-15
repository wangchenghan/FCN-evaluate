# FCN-evaluate
用于FCN结果评估的代码，包含pixel accuarcy,mean accuarcy,mean IU,frequency weighted IU

需要numpy和PIL

在用于评价的文件夹下需要有名为‘gt’和‘pred’两个文件夹,其中‘gt’包含标注图片,‘pred’包含FCN输出的预测结果。改一下主函数中dir1的地址就能用了。
