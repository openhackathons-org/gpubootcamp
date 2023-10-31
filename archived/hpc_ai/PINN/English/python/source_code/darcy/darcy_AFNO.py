import modulus
from modulus.hydra import instantiate_arch
from modulus.hydra.config import ModulusConfig
from modulus.key import Key

from modulus.domain import Domain
from modulus.domain.constraint import SupervisedGridConstraint
from modulus.domain.validator import GridValidator
from modulus.dataset import DictGridDataset
from modulus.solver import Solver

from modulus.utils.io.plotter import GridValidatorPlotter

from utilities import download_FNO_dataset, load_FNO_dataset


@modulus.main(config_path="conf", config_name="config_AFNO")
def run(cfg: ModulusConfig) -> None:

    # load training/ test data
    input_keys = [Key("coeff", scale=(7.48360e00, 4.49996e00))]
    output_keys = [Key("sol", scale=(5.74634e-03, 3.88433e-03))]

    download_FNO_dataset("Darcy_241", outdir="datasets/")
    invar_train, outvar_train = load_FNO_dataset(
        "datasets/Darcy_241/piececonst_r241_N1024_smooth1.hdf5",
        [k.name for k in input_keys],
        [k.name for k in output_keys],
        n_examples=1000,
    )
    invar_test, outvar_test = load_FNO_dataset(
        "datasets/Darcy_241/piececonst_r241_N1024_smooth2.hdf5",
        [k.name for k in input_keys],
        [k.name for k in output_keys],
        n_examples=100,
    )

    # get training image shape
    img_shape = [
        next(iter(invar_train.values())).shape[-2],
        next(iter(invar_train.values())).shape[-1],
    ]

    # crop out some pixels so that img_shape is divisible by patch_size of AFNO
    img_shape = [s - s % cfg.arch.afno.patch_size for s in img_shape]
    print(f"cropped img_shape: {img_shape}")
    for d in (invar_train, outvar_train, invar_test, outvar_test):
        for k in d:
            d[k] = d[k][:, :, : img_shape[0], : img_shape[1]]
            print(f"{k}: {d[k].shape}")

    # make datasets
    train_dataset = DictGridDataset(invar_train, outvar_train)
    test_dataset = DictGridDataset(invar_test, outvar_test)

    # make list of nodes to unroll graph on
    model = instantiate_arch(
        input_keys=input_keys,
        output_keys=output_keys,
        cfg=cfg.arch.afno,
        img_shape=img_shape,
    )
    nodes = [model.make_node(name="AFNO", jit=cfg.jit)]

    # make domain
    domain = Domain()

    # add constraints to domain
    supervised = SupervisedGridConstraint(
        nodes=nodes,
        dataset=train_dataset,
        batch_size=cfg.batch_size.grid,
    )
    domain.add_constraint(supervised, "supervised")

    # add validator
    val = GridValidator(
        nodes,
        dataset=test_dataset,
        batch_size=cfg.batch_size.validation,
        plotter=GridValidatorPlotter(n_examples=5),
    )
    domain.add_validator(val, "test")

    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()
