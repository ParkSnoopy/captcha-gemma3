# file: main.py
import os
import sys
import shlex
import argparse

import tensorflow as tf
from PIL import Image
import numpy as np


def arg_parse():
    parser = argparse.ArgumentParser()
    # image spec
    parser.add_argument("--image_width", type=int, default=160)
    parser.add_argument("--image_height", type=int, default=60)
    parser.add_argument("--image_channel", type=int, default=1)
    # text spec
    parser.add_argument("--text_len", type=int, default=5)
    parser.add_argument(
        "--char_set", type=str, default="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    )
    # train config
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epoch_num", type=int, default=10)
    parser.add_argument("--learn_rate", type=float, default=1e-3)
    # data path
    parser.add_argument("--train_csv", type=str, default="")
    parser.add_argument("--val_csv", type=str, default="")
    # io path
    parser.add_argument("--save_path", type=str, default="model.keras")
    parser.add_argument("--load_path", type=str, default="")
    # infer
    parser.add_argument("--infer_path", type=str, default="")
    # misc
    parser.add_argument("--seed_num", type=int, default=42)

    # 코드 내에서 문자열 인자를 지정하고 싶다면 여기 수정
    code_arg = (
        ""  # 예: ' --train_csv data/train.csv --val_csv data/val.csv --epoch_num 20 '
    )

    # 1) 코드 문자열 기반 파싱
    if code_arg and code_arg.strip():
        args = parser.parse_args(shlex.split(code_arg))
    else:
        args = parser.parse_args([])  # defaults

    # 2) 터미널 인자로 override (우선순위 최상)
    args = parser.parse_args(namespace=args)
    return args


def data_make(
    csv_path,
    image_width,
    image_height,
    image_channel,
    text_len,
    char_set,
    batch_size,
    shuffle_flag,
):
    if not csv_path:
        return None
    char_to_id = {c: i for i, c in enumerate(char_set)}
    id_to_pad = (
        0  # 왜: 간단화를 위해 패딩을 첫 글자에 매핑(실데이터는 text_len 맞추길 권장)
    )

    image_list, label_list = [], []
    with open(csv_path, "r", encoding="utf-8") as file_obj:
        for line in file_obj:
            line = line.strip()
            if not line or "," not in line:
                continue
            path_part, label_part = line.split(",", 1)
            path_part = path_part.strip()
            label_part = label_part.strip()
            if not os.path.isfile(path_part):
                continue
            # 길이 불일치 시 스킵(학습 안정성)
            if len(label_part) != text_len:
                continue

            try:
                with Image.open(path_part) as image_obj:
                    if image_channel == 1:
                        image_obj = image_obj.convert("L")
                    else:
                        image_obj = image_obj.convert("RGB")
                    image_obj = image_obj.resize((image_width, image_height))
                    image_arr = np.asarray(image_obj, dtype=np.float32)
            except Exception:
                continue

            if image_channel == 1:
                if image_arr.ndim == 2:
                    image_arr = np.expand_dims(image_arr, axis=-1)
            image_arr = image_arr / 255.0
            image_list.append(image_arr)

            ids = [char_to_id.get(ch, id_to_pad) for ch in label_part]
            label_list.append(ids)

    if not image_list:
        return None

    image_arr = np.stack(image_list, axis=0)
    label_arr = np.array(label_list, dtype=np.int32)

    dataset = tf.data.Dataset.from_tensor_slices((image_arr, label_arr))
    if shuffle_flag:
        dataset = dataset.shuffle(
            buffer_size=min(len(image_list), 10000),
            seed=42,
            reshuffle_each_iteration=True,
        )
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def model_build(
    image_width, image_height, image_channel, text_len, char_set, learn_rate
):
    vocab_num = len(char_set)
    input_obj = tf.keras.Input(
        shape=(image_height, image_width, image_channel), name="image_input"
    )

    x = tf.keras.layers.Conv2D(32, 3, padding="same")(input_obj)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)

    x = tf.keras.layers.Conv2D(64, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)

    x = tf.keras.layers.Conv2D(128, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.Dense(text_len * vocab_num)(x)
    x = tf.keras.layers.Reshape((text_len, vocab_num))(x)
    out = tf.keras.layers.Softmax(axis=-1, name="text_output")(x)

    model_obj = tf.keras.Model(inputs=input_obj, outputs=out)
    # 왜: 위치별 독립 다중분류로 단순화(CTC 미사용)
    model_obj.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learn_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="char_acc")],
    )
    return model_obj


def infer_run(
    model_obj, image_path, image_width, image_height, image_channel, char_set
):
    with Image.open(image_path) as image_obj:
        if image_channel == 1:
            image_obj = image_obj.convert("L")
        else:
            image_obj = image_obj.convert("RGB")
        image_obj = image_obj.resize((image_width, image_height))
        image_arr = np.asarray(image_obj, dtype=np.float32)
        if image_channel == 1 and image_arr.ndim == 2:
            image_arr = np.expand_dims(image_arr, axis=-1)
        image_arr = image_arr / 255.0
    batch_arr = np.expand_dims(image_arr, axis=0)
    pred_arr = model_obj.predict(batch_arr, verbose=0)[0]  # (text_len, vocab)
    idx_arr = np.argmax(pred_arr, axis=-1)
    char_out = "".join(char_set[i] for i in idx_arr)
    print(char_out)


def main():
    args = arg_parse()
    tf.keras.utils.set_random_seed(args.seed_num)

    train_set = data_make(
        args.train_csv,
        args.image_width,
        args.image_height,
        args.image_channel,
        args.text_len,
        args.char_set,
        args.batch_size,
        shuffle_flag=True,
    )
    val_set = data_make(
        args.val_csv,
        args.image_width,
        args.image_height,
        args.image_channel,
        args.text_len,
        args.char_set,
        args.batch_size,
        shuffle_flag=False,
    )

    model_obj = model_build(
        args.image_width,
        args.image_height,
        args.image_channel,
        args.text_len,
        args.char_set,
        args.learn_rate,
    )

    if args.load_path and os.path.isfile(args.load_path):
        model_obj.load_weights(args.load_path)

    if train_set is not None:
        callbacks_list = (
            [
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=args.save_path,
                    save_weights_only=False,
                    save_best_only=True,
                    monitor="val_char_acc",
                    mode="max",
                )
            ]
            if val_set is not None
            else []
        )
        model_obj.fit(
            train_set,
            validation_data=val_set,
            epochs=args.epoch_num,
            callbacks=callbacks_list,
            verbose=1,
        )
        # 마지막 가중치 저장(최적이 아니어도 기록)
        try:
            model_obj.save(args.save_path)
        except Exception:
            pass

    if args.infer_path:
        if os.path.isfile(args.infer_path):
            infer_run(
                model_obj,
                args.infer_path,
                args.image_width,
                args.image_height,
                args.image_channel,
                args.char_set,
            )
        else:
            print("infer_path not found", file=sys.stderr)


if __name__ == "__main__":
    main()
