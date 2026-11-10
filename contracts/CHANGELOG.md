# Contract changelog

## 1.1.0 — M1 (совместимое добавление)
- `LawMeta.article_title`, `LawMeta.alias` (оба необязательные): заголовок статьи и короткое имя акта из имени файла — нужны индексу и рендеру ссылок.

## 1.0.0 — M0
Первая версия контрактов (ARCHITECTURE §5) с дополнениями по итогам разбора базы:
- `LawAmendmentNote.scope` (`ч. 1`, `п. 3`, `пп. 2`; None = вся статья) и `repealed` — пометки об изменениях в законах стоят и в шапке, и перед частями/пунктами/подпунктами.
- `DocCard.approx_tokens`, `KnowledgeBase.article_inventory`, `Chunk.is_editorial` — с дефолтами.
- `ChatRequest.last_user_text()` — вспомогательный метод.
