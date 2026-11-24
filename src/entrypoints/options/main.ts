import { mount } from "svelte";
import { applyDocumentDirection } from "../../shared/i18n";
import App from "./App.svelte";
import "./style.css";

applyDocumentDirection();

const target = document.getElementById("app");
if (target == null) throw new Error("options mount target missing");
mount(App, { target });
