Linear/GVP/VP mode by the following path
1. args [mdgen.parsing](../mdgen/parsing.py)
    >```python
    >group.add_argument("--path-type", type=str, default="GVP", choices=["Linear", "GVP", "VP"])
    >```
1. NewMDGenWrapper [train.py](../train.py)
    >```python
    >model = NewMDGenWrapper(args)
    >```
1. create_transport [mdgen.wrapper](../mdgen/wrapper.py)
    >```python
    >class NewMDGenWrapper(Wrapper):
    >...
    >   self.transport = create_transport(
    >       args,
    >       args.path_type,
    >       args.prediction,
    >       None,  # args.loss_weight,
    >       # args.train_eps,
    >       # args.sample_eps,
    >   )  # default: velocity; 
    >```
1. Transport [mdgen.transport.transport.py](../mdgen/transport/transport.py)
    >```python
    >def create_transport(
    >        args,
    >        path_type='Linear',
    >        prediction="velocity",
    >        loss_weight=None,
    >        train_eps=None,
    >        sample_eps=None,
    >):
    >...
    >    # create flow state
    >    state = Transport(
    >        args=args,
    >        model_type=model_type,
    >        path_type=path_type,
    >        loss_type=loss_type,
    >        train_eps=train_eps,
    >        sample_eps=sample_eps,
    >    )
    >```
1. path_opthions [mdgen.transport.transport.py](../mdgen/transport/transport.py)
    >```python
    >class Transport:
    >
    >    def __init__(
    >            self,
    >            *,
    >            args,
    >            model_type,
    >            path_type,
    >            loss_type,
    >            train_eps,
    >            sample_eps,
    >    ):
    >    ...
    >    self.path_sampler = path_options[path_type]()
    >    ...
    >    def training_losses(
    >            self,
    >            model,
    >            x1,           # target tokens
    >            aatype1=None, # target aatype
    >            mask=None,
    >            model_kwargs=None
    >    ):
    >        """Loss for training the score model
    >        Args:
    >        - model: backbone model; could be score, noise, or velocity
    >        - x1: datapoint
    >        - model_kwargs: additional arguments for the model
    >        """
    >
    >        if model_kwargs == None:
    >            model_kwargs = {}
    >
    >        t, x0, x1 = self.sample(x1)
    >        t, xt, ut = self.path_sampler.plan(t, x0, x1)
    >```
1. PathType,ICPlan,take linear as example [mdgen.transport.transport.py](../mdgen/transport/transport.py)
    >```python
    >path_options = {
    >    PathType.LINEAR: path.ICPlan,
    >    PathType.GVP: path.GVPCPlan,
    >    PathType.VP: path.VPCPlan,
    >}
    >```
1. end [mdgen.transport.path.py](../mdgen/transport/path.py)
    >```python
    >class ICPlan:
    >    """Linear Coupling Plan"""
    >    def __init__(self, sigma=0.0):
    >        self.sigma = sigma
    >
    >    def compute_alpha_t(self, t):
    >        """Compute the data coefficient along the path"""
    >        return t, 1
    >...
    >```