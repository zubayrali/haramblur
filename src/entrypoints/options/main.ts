import { mount } from "svelte";
import App from "./App.svelte";
import "./style.css";

const target = document.getElementById("app");
if (target == null) throw new Error("options mount target missing");
mount(App, { target });
