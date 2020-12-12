# optimizer
Some optimizer algorithms implemented using pytorch 

```python
# optimizer = SGD(net.parameters(), lr=1e-2)
# optimizer = MSGD(net.parameters(), lr=0.001, momentum=0.9)
# optimizer = Nesterov(net.parameters(), lr=0.001, momentum=0.9)
# optimizer = AdaGrad(net.parameters())
# optimizer = AdaDelta(net.parameters())
# optimizer = AdaDelta(net.parameters(), lr= 0.001)
# optimizer = RMSProp(net.parameters(), lr= 0.001)
# optimizer = Adam(net.parameters(), lr= 0.001)
# optimizer = Nadam(net.parameters(), lr= 0.001)
# optimizer = ASGD(net.parameters())
# optimizer = SAG(net.parameters())
# optimizer = SVRG(net.parameters(), batch_size = batch_size, epoch = 5)
# optimizer = MirrorDescent(net.parameters(), lr = 0.01, BreDivFun ='Squared norm')
# optimizer = MDNesterov(net.parameters(), lr = 0.01, momentum = 0.8, BreDivFun ='Squared norm')
```
