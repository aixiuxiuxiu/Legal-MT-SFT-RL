from dataclasses import dataclass

from PIL import Image

from dataset.resize import ResizeWhen, resize_larger_side


@dataclass
class ResizeTestCase:
    size: tuple[int, int]
    when: ResizeWhen
    larger_side: int
    expected_larger_side: tuple[int, int]

    def assert_resize_larger_side(self):
        img = Image.new("RGB", size=self.size)
        assert img.size == self.size
        img_resized = resize_larger_side(img, size=self.larger_side, when=self.when)
        assert img_resized.size == self.expected_larger_side


TEST_CASES = [
    ResizeTestCase(
        size=(768, 1024),
        when="always",
        larger_side=512,
        expected_larger_side=(384, 512),
    ),
    ResizeTestCase(
        size=(1024, 768),
        when="always",
        larger_side=512,
        expected_larger_side=(512, 384),
    ),
    ResizeTestCase(
        size=(1024, 1024),
        when="always",
        larger_side=512,
        expected_larger_side=(512, 512),
    ),
    ResizeTestCase(
        size=(768, 1024),
        when="never",
        larger_side=512,
        expected_larger_side=(768, 1024),
    ),
    ResizeTestCase(
        size=(768, 1024),
        when="when-smaller",
        larger_side=512,
        expected_larger_side=(768, 1024),
    ),
    ResizeTestCase(
        size=(128, 256),
        when="when-smaller",
        larger_side=512,
        expected_larger_side=(256, 512),
    ),
    ResizeTestCase(
        size=(256, 128),
        when="when-smaller",
        larger_side=512,
        expected_larger_side=(512, 256),
    ),
    ResizeTestCase(
        size=(128, 1024),
        when="when-smaller",
        larger_side=512,
        expected_larger_side=(128, 1024),
    ),
    ResizeTestCase(
        size=(1024, 128),
        when="when-smaller",
        larger_side=512,
        expected_larger_side=(1024, 128),
    ),
    ResizeTestCase(
        size=(768, 1024),
        when="when-larger",
        larger_side=512,
        expected_larger_side=(384, 512),
    ),
    ResizeTestCase(
        size=(1024, 768),
        when="when-larger",
        larger_side=512,
        expected_larger_side=(512, 384),
    ),
    ResizeTestCase(
        size=(128, 256),
        when="when-larger",
        larger_side=512,
        expected_larger_side=(128, 256),
    ),
    ResizeTestCase(
        size=(128, 1024),
        when="when-larger",
        larger_side=512,
        expected_larger_side=(64, 512),
    ),
    ResizeTestCase(
        size=(1024, 128),
        when="when-larger",
        larger_side=512,
        expected_larger_side=(512, 64),
    ),
]


def test_resize_larger_side():
    for test_case in TEST_CASES:
        test_case.assert_resize_larger_side()
