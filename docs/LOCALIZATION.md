# Localization and RTL

## Architecture

`I18nProvider` supplies `language`, `setLanguage`, `t` and direction. Resources are typed from the English tree and live in:

- `frontend/src/locales/en/common.ts`
- `frontend/src/locales/ar/common.ts`

`t(key, variables)` resolves dotted keys, interpolates `{{name}}` values and falls back to English. The language setting is persisted both in player settings (`PATCH /api/p/{pid}/settings`) and browser storage for startup before the player request completes.

Changing locale updates:

```html
<html lang="ar" dir="rtl">
```

or its English equivalent without reloading.

## RTL behavior

`rtl-and-labs.css` uses logical properties and explicit RTL overrides for:

- fixed navigation and main-content offset;
- rows, forms, cards, tabs, tables and action groups;
- mixed Arabic/English text;
- responsive mobile navigation;
- chart labels and overflow surfaces.

Technical content remains LTR through `Ltr`, `.technical-ltr`, `code`, `pre`, model identifiers, equations, JSON, file names, metrics and charts. This prevents punctuation/number corruption in Arabic layouts.

## Typography

The font stack uses Arabic-capable system fonts (`Noto Sans Arabic`, `Segoe UI`, Tahoma) before generic fallbacks and increases Arabic line height. There is no remote font dependency.

## Resource rules

1. Do not branch on `language === 'ar'` inside components.
2. Add the same dotted key to both resource trees.
3. Keep model names, code, commands and units as LTR values.
4. Translate labels and explanatory prose, not serialized API identifiers.
5. Use interpolation for values instead of concatenating translated fragments.
6. Test resource-key parity and DOM `lang`/`dir` persistence.

## Current coverage

The application shell, onboarding, campus summary, settings and all newly added personal-lab surfaces use structured English/Arabic resources. RTL layout applies globally, including preserved campaign/lab pages. Some preserved legacy campaign widgets and long pedagogical prose remain authored directly in English; they render safely but are not yet fully translated. This is tracked as **partial localization**, not described as complete Arabic coverage, in the capability matrix.

## Tests

`frontend/tests/localization.test.tsx` verifies:

- exact key parity between English and Arabic resources;
- persisted Arabic startup;
- document `lang`/`dir` changes;
- technical content remains LTR;
- switching persists the new locale.
