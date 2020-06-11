from embedded_model.object_detection import model_lib
import tensorflow as tf

class FineTuneAPI:

    _EMBEDDED_MODEL_CHECKPOINT = "~/.nglite/pretrained"
    _EMBEDDED_OBJECTDETECTION_NAME = "ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03"

    def finetune_embedded_objectdetection_model(self, pipeline_config_path, target_dir, train_steps=20000):

        config = tf.estimator.RunConfig(model_dir=target_dir)

        train_and_eval_dict = model_lib.create_estimator_and_inputs(
            run_config=config,
            hparams=self._create_hparams(None),
            pipeline_config_path=pipeline_config_path,
            train_steps=train_steps,
            sample_1_of_n_eval_examples=1,
            sample_1_of_n_eval_on_train_examples=5)

        estimator = train_and_eval_dict['estimator']
        train_input_fn = train_and_eval_dict['train_input_fn']
        eval_input_fns = train_and_eval_dict['eval_input_fns']
        eval_on_train_input_fn = train_and_eval_dict['eval_on_train_input_fn']
        predict_input_fn = train_and_eval_dict['predict_input_fn']
        train_steps = train_and_eval_dict['train_steps']

        train_spec, eval_specs = model_lib.create_train_and_eval_specs(
            train_input_fn,
            eval_input_fns,
            eval_on_train_input_fn,
            predict_input_fn,
            train_steps,
            eval_on_train_data=False)
        # Currently only a single Eval Spec is allowed.
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_specs[0])

    def _create_hparams(self, hparams_overrides=None):
        """Returns hyperparameters, including any flag value overrides.

        Args:
          hparams_overrides: Optional hparams overrides, represented as a
            string containing comma-separated hparam_name=value pairs.

        Returns:
          The hyperparameters as a tf.HParams object.
        """
        hparams = tf.contrib.training.HParams(
            # Whether a fine tuning checkpoint (provided in the pipeline config)
            # should be loaded for training.
            load_pretrained=True)
        # Override any of the preceding hyperparameter values.
        if hparams_overrides:
            hparams = hparams.parse(hparams_overrides)
        return hparams