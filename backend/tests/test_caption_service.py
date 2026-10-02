from PIL import Image

from app.services import caption_service


def test_caption_service_normalizes_tiny_grayscale_images(tmp_path, monkeypatch):
    path = tmp_path / 'tiny.png'
    Image.new('L', (1, 1), color=128).save(path)
    received = {}

    def captioner(image, max_new_tokens):
        received.update({'image': image, 'max_new_tokens': max_new_tokens})
        return [{'generated_text': 'A small gray image'}]

    monkeypatch.setattr(caption_service, 'get_caption_pipeline', lambda: captioner)

    result = caption_service.generate_caption(str(path))

    assert result == {'caption': 'A small gray image', 'status': 'completed', 'error': None}
    assert received['image'].mode == 'RGB'
    assert received['image'].size == (32, 32)


def test_caption_service_reports_empty_model_output(tmp_path, monkeypatch):
    path = tmp_path / 'blank.png'
    Image.new('RGB', (32, 32), color='white').save(path)
    monkeypatch.setattr(
        caption_service,
        'get_caption_pipeline',
        lambda: lambda *_args, **_kwargs: [{'generated_text': ''}],
    )

    result = caption_service.generate_caption(str(path))

    assert result['caption'] == ''
    assert result['status'] == 'failed'
    assert result['error'] == 'The caption model returned no text for this image.'
