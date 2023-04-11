## Image Style Transfer


### Problem : Require_grad = True
Autograd of torch.Tensor
* .requires_grad = True
  * track the results of the tensor
* .detach()
  * detach the tensor of tracking
* with torch.no_grad()
  *  requires_grad = True automatically
* grad_fn attribute
  * tensor which is made by user-> grad_fn = None
  * tensor which is result of the calculation  
  -> grad_fn reference is Function class which make the Tensor
* Tensor and Function are connected and encode every calculation process
