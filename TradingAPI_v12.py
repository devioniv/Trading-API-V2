import sys
import asyncio
import os
import json
import time
import traceback
import inspect
from typing import Optional, Dict, Any, List

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTabWidget, QLabel, QLineEdit, QComboBox,
    QTextEdit, QGroupBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QMessageBox, QDoubleSpinBox, QAbstractItemView, QCheckBox, QSpinBox
)
from PySide6.QtCore import Signal, Qt

try:
    from ib_async import IB, util, Stock, Forex, Future, Option, Crypto, MarketOrder, LimitOrder, StopOrder
    from numpy.random import default_rng
except Exception:
    IB = None
    MarketOrder = None
    LimitOrder = None
    StopOrder = None
    default_rng = None

import pandas as pd
import pandas_ta as ta
import qasync
import numpy as np

CONFIG_FILE = "trading_config.json"

def normalize_ta_columns(df: pd.DataFrame):
    import re
    try:
        if df is None:
            return
        if "ATRr_14" in df.columns and "ATR_14" not in df.columns:
            df.rename(columns={"ATRr_14": "ATR_14"}, inplace=True)

        adx_col = next((c for c in df.columns if c.startswith('ADX_')), None)
        if adx_col and adx_col != 'ADX':
            df.rename(columns={adx_col: 'ADX'}, inplace=True)

        macd_candidates = [c for c in df.columns if re.search(r'(?i)macd', c)]
        main = signal = hist = None
        for c in macd_candidates:
            lc = c.lower()
            if lc == "macd" or (lc.startswith("macd_") and main is None):
                main = c
            if "signal" in lc and signal is None:
                signal = c
            if "hist" in lc and hist is None:
                hist = c
        renames = {}
        if main and main != "MACD": renames[main] = "MACD"
        if signal and signal != "MACD_signal": renames[signal] = "MACD_signal"
        if hist and hist != "MACD_hist": renames[hist] = "MACD_hist"
        if renames:
            df.rename(columns=renames, inplace=True)
    except Exception:
        pass

class TradingApp(QMainWindow):
    connection_status_signal = Signal(bool)
    account_update_signal = Signal(str, str, str)
    position_update_signal = Signal(str, float, float)
    instrument_list_update_signal = Signal(list)
    log_signal = Signal(str)
    summary_signal = Signal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("TradingAPI v12")
        self.resize(1400, 920)

        self.ib = IB() if IB is not None else None

        if self.ib:
            self.ib.barUpdateEvent.connect(self._on_bar_update)
            for event_name in ("connectedEvent", "disconnectedEvent", "orderStatusEvent",
                               "execDetailsEvent", "errorEvent", "accountValueEvent", "positionEvent"):
                try:
                    evt = getattr(self.ib, event_name)
                    if event_name == "connectedEvent": evt += self.on_connected
                    elif event_name == "disconnectedEvent": evt += self.on_disconnected
                    elif event_name == "orderStatusEvent": evt += self.on_order_status
                    elif event_name == "execDetailsEvent": evt += self.on_exec_details
                    elif event_name == "errorEvent": evt += self.on_ib_error
                    elif event_name == "accountValueEvent": evt += self.on_account_value_event
                    elif event_name == "positionEvent": evt += self.on_position_event
                except Exception:
                    pass

        self.tasks = set()
        self.instruments: Dict[str, Dict[str, Any]] = {}
        self.data: Dict[str, pd.DataFrame] = {}
        self.strategies: Dict[int, object] = {}
        self.strategy_tasks: Dict[int, asyncio.Task] = {}
        self.next_strategy_id = 1
        self.live_bar_subscriptions: Dict[str, object] = {}

        self.log_signal.connect(self.append_log)
        self.connection_status_signal.connect(self.update_connection_status)
        self.account_update_signal.connect(self.update_account_table)
        self.position_update_signal.connect(self.update_positions_table)
        self.instrument_list_update_signal.connect(self.update_instrument_selectors)
        self.summary_signal.connect(self.update_summary_text)

        self.create_widgets()
        self.load_config()

    def create_widgets(self):
        central = QWidget()
        self.setCentralWidget(central)
        v = QVBoxLayout(central)

        top = QHBoxLayout()
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self.connect_ib)
        self.disconnect_btn = QPushButton("Disconnect")
        self.disconnect_btn.clicked.connect(self.disconnect_ib)
        top.addWidget(self.connect_btn)
        top.addWidget(self.disconnect_btn)
        top.addSpacing(24)
        top.addWidget(QLabel("Symbol:"))
        self.symbol_entry = QLineEdit()
        self.symbol_entry.setFixedWidth(120)
        top.addWidget(self.symbol_entry)
        top.addWidget(QLabel("SecType:"))
        self.sec_type = QComboBox()
        self.sec_type.addItems(["STK", "CASH", "FUT", "OPT", "CRYPTO"])
        top.addWidget(self.sec_type)
        self.add_inst_btn = QPushButton("Add Instrument")
        self.add_inst_btn.clicked.connect(self.on_add_instrument)
        top.addWidget(self.add_inst_btn)
        v.addLayout(top)

        self.tabs = QTabWidget()
        v.addWidget(self.tabs)

        self.tab_dash = QWidget()
        self.tabs.addTab(self.tab_dash, "Dashboard")
        dash_layout = QHBoxLayout(self.tab_dash)
        left = QVBoxLayout()
        self.account_table = QTableWidget(0, 3)
        self.account_table.setHorizontalHeaderLabels(["Tag", "Value", "Currency"])
        self._set_header_resize(self.account_table, 3)
        left.addWidget(QLabel("Account"))
        left.addWidget(self.account_table)
        self.positions_table = QTableWidget(0, 3)
        self.positions_table.setHorizontalHeaderLabels(["Symbol", "Quantity", "Avg Cost"])
        self._set_header_resize(self.positions_table, 3)
        left.addWidget(QLabel("Positions"))
        left.addWidget(self.positions_table)
        dash_layout.addLayout(left, 2)
        right = QVBoxLayout()
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        right.addWidget(QLabel("Summary"))
        right.addWidget(self.summary_text)
        dash_layout.addLayout(right, 1)

        self.tab_inst = QWidget()
        self.tabs.addTab(self.tab_inst, "Instruments")
        inst_layout = QVBoxLayout(self.tab_inst)
        self.inst_table = QTableWidget(0, 3)
        self.inst_table.setHorizontalHeaderLabels(["Symbol", "SecType", "DataType"])
        self._set_header_resize(self.inst_table, 3)
        self.inst_table.setSelectionBehavior(getattr(QAbstractItemView, "SelectRows", QAbstractItemView.SelectionBehavior.SelectRows))
        self.inst_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        inst_layout.addWidget(self.inst_table)
        inst_btns = QHBoxLayout()
        self.remove_inst_btn = QPushButton("Remove Selected")
        self.remove_inst_btn.clicked.connect(self.on_remove_selected_instruments)
        inst_btns.addWidget(self.remove_inst_btn)
        inst_layout.addLayout(inst_btns)

        self.tab_strat = QWidget()
        self.tabs.addTab(self.tab_strat, "Strategy Manager")
        strat_layout = QVBoxLayout(self.tab_strat)

        top_strat_row = QHBoxLayout()
        self.select_all_chk = QCheckBox("Select All Strategies")
        self.select_all_chk.stateChanged.connect(self.on_select_all_toggled)
        top_strat_row.addWidget(self.select_all_chk)
        top_strat_row.addStretch()
        strat_layout.addLayout(top_strat_row)

        self.strategy_types = ["WMA_RMA_MACD", "TrendlineBreakout", "SupportResistance", "FairValueGap"]
        self.strategy_checkboxes: Dict[str, QCheckBox] = {}
        strat_box = QGroupBox("Available Strategies")
        strat_box_layout = QHBoxLayout()
        for s in self.strategy_types:
            cb = QCheckBox(s)
            cb.setChecked(True)
            cb.stateChanged.connect(self._on_individual_strategy_toggled)
            self.strategy_checkboxes[s] = cb
            strat_box_layout.addWidget(cb)
        strat_box.setLayout(strat_box_layout)
        strat_layout.addWidget(strat_box)

        params_group = QGroupBox("Strategy-Specific Parameters")
        params_v_layout = QVBoxLayout()
        wma_macd_box = QGroupBox("WMA_RMA_MACD")
        wma_macd_layout = QHBoxLayout()
        wma_macd_layout.addWidget(QLabel("WMA Fast:"))
        self.wma_fast_spin = QSpinBox()
        self.wma_fast_spin.setRange(1, 1000); self.wma_fast_spin.setValue(50)
        wma_macd_layout.addWidget(self.wma_fast_spin)
        wma_macd_layout.addWidget(QLabel("RMA Slow:"))
        self.rma_slow_spin = QSpinBox()
        self.rma_slow_spin.setRange(1, 1000); self.rma_slow_spin.setValue(100)
        wma_macd_layout.addWidget(self.rma_slow_spin)
        wma_macd_layout.addWidget(QLabel("MACD Fast:"))
        self.macd_fast_spin = QSpinBox()
        self.macd_fast_spin.setRange(1, 1000); self.macd_fast_spin.setValue(5)
        wma_macd_layout.addWidget(self.macd_fast_spin)
        wma_macd_layout.addWidget(QLabel("MACD Slow:"))
        self.macd_slow_spin = QSpinBox()
        self.macd_slow_spin.setRange(1, 1000); self.macd_slow_spin.setValue(10)
        wma_macd_layout.addWidget(self.macd_slow_spin)
        wma_macd_layout.addWidget(QLabel("MACD Signal:"))
        self.macd_signal_spin = QSpinBox()
        self.macd_signal_spin.setRange(1, 1000); self.macd_signal_spin.setValue(10)
        wma_macd_layout.addWidget(self.macd_signal_spin)
        wma_macd_layout.addStretch()
        wma_macd_box.setLayout(wma_macd_layout)
        params_v_layout.addWidget(wma_macd_box)

        fvg_box = QGroupBox("FairValueGap")
        fvg_layout = QVBoxLayout()
        fvg_row1 = QHBoxLayout()
        fvg_row1.addWidget(QLabel("Min Gap Size:"))
        self.fvg_min_gap_spin = QDoubleSpinBox()
        self.fvg_min_gap_spin.setRange(0.0, 10.0); self.fvg_min_gap_spin.setSingleStep(0.1); self.fvg_min_gap_spin.setValue(0.1)
        fvg_row1.addWidget(self.fvg_min_gap_spin)
        fvg_row1.addWidget(QLabel("ATR Mult (for SL):"))
        self.fvg_trail_atr_mult_spin = QDoubleSpinBox()
        self.fvg_trail_atr_mult_spin.setRange(0.1, 10.0); self.fvg_trail_atr_mult_spin.setSingleStep(0.1); self.fvg_trail_atr_mult_spin.setValue(1.0)
        fvg_row1.addWidget(self.fvg_trail_atr_mult_spin)
        fvg_row1.addStretch()
        fvg_layout.addLayout(fvg_row1)
        fvg_row2 = QHBoxLayout()
        self.fvg_dynamic_sizing_chk = QCheckBox("Dynamic Sizing")
        self.fvg_dynamic_sizing_chk.setChecked(True)
        fvg_row2.addWidget(self.fvg_dynamic_sizing_chk)
        self.fvg_partial_tp_chk = QCheckBox("Partial TP")
        self.fvg_partial_tp_chk.setChecked(True)
        fvg_row2.addWidget(self.fvg_partial_tp_chk)
        self.fvg_trailing_stop_chk = QCheckBox("Trailing Stop")
        self.fvg_trailing_stop_chk.setChecked(True)
        fvg_row2.addWidget(self.fvg_trailing_stop_chk)
        fvg_row2.addStretch()
        fvg_layout.addLayout(fvg_row2)
        fvg_box.setLayout(fvg_layout)
        params_v_layout.addWidget(fvg_box)

        params_group.setLayout(params_v_layout)
        strat_layout.addWidget(params_group)

        params_row = QHBoxLayout()
        self.apply_to_selected_chk = QCheckBox("Apply to selected instruments")
        self.apply_to_selected_chk.setChecked(True)
        params_row.addWidget(self.apply_to_selected_chk)
        params_row.addWidget(QLabel("Single Symbol:"))
        self.single_symbol_dropdown = QComboBox()
        params_row.addWidget(self.single_symbol_dropdown)
        params_row.addWidget(QLabel("Qty:"))
        self.qty_spin = QDoubleSpinBox()
        self.qty_spin.setValue(1.0)
        params_row.addWidget(self.qty_spin)
        params_row.addWidget(QLabel("Interval(s):"))
        self.interval_spin = QDoubleSpinBox()
        self.interval_spin.setValue(5.0)
        params_row.addWidget(self.interval_spin)
        self.auto_trade_chk = QCheckBox("Auto Trade (LIVE)")
        params_row.addWidget(self.auto_trade_chk)
        self.create_checked_btn = QPushButton("Create + Start Checked Strategies")
        self.create_checked_btn.clicked.connect(self.on_create_checked_strategies)
        params_row.addWidget(self.create_checked_btn)
        strat_layout.addLayout(params_row)

        self.running_strats_table = QTableWidget(0, 6)
        self.running_strats_table.setHorizontalHeaderLabels(["ID", "Symbol", "Strategy", "Qty", "Interval", "Action"])
        self._set_header_resize(self.running_strats_table, 6)
        strat_layout.addWidget(self.running_strats_table)

        self.tab_logs = QWidget()
        self.tabs.addTab(self.tab_logs, "Logs")
        logs_layout = QVBoxLayout(self.tab_logs)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        logs_layout.addWidget(self.log_text)

        self.statusBar().showMessage("Ready")

    def _set_header_resize(self, table: QTableWidget, cols: int):
        try:
            header = table.horizontalHeader()
            mode = getattr(QHeaderView, "Stretch", getattr(QHeaderView, "ResizeMode", QHeaderView).Stretch)
            for c in range(cols):
                header.setSectionResizeMode(c, mode)
        except Exception:
            pass

    def add_task(self, coro):
        try:
            task = asyncio.create_task(coro)
            self.tasks.add(task)
            task.add_done_callback(self.tasks.discard)
            return task
        except Exception as e:
            self.log(f"add_task error: {e}")
            return None

    def append_log(self, msg: str):
        try:
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            self.log_text.append(f"[{ts}] {msg}")
        except Exception:
            pass

    def log(self, msg: str):
        print(f"LOG: {msg}")
        self.log_signal.emit(str(msg))

    def connect_ib(self):
        self.log("Attempting IB connect...")
        if not self.ib:
            self.log("IB library not available.")
            return
        self.add_task(self._connect_async())

    async def _connect_async(self):
        try:
            await self.ib.connectAsync("127.0.0.1", 7497, clientId=1)
        except Exception as e:
            self.log(f"IB connect failed: {e}")

    def disconnect_ib(self):
        if self.ib:
            self.ib.disconnect()

    def on_connected(self):
        self.connection_status_signal.emit(True)
        if not self.ib: return
        try:
            self.add_task(self.ib.reqAccountUpdatesAsync(True, ""))
        except Exception as e:
            self.log(f"reqAccountUpdatesAsync failed: {e}")

    def on_disconnected(self):
        self.connection_status_signal.emit(False)

    def update_connection_status(self, is_connected: bool):
        self.connect_btn.setEnabled(not is_connected)
        self.disconnect_btn.setEnabled(is_connected)
        self.statusBar().showMessage("Connected to IB." if is_connected else "Disconnected.")

    def on_position_event(self, pos):
        sym = getattr(getattr(pos, "contract", None), "symbol", None)
        if sym:
            self.position_update_signal.emit(sym, float(pos.position), float(pos.avgCost))

    def on_account_value_event(self, val):
        if val.tag in ["NetLiquidation", "TotalCashValue", "UnrealizedPnL", "RealizedPnL", "BuyingPower"]:
            self.account_update_signal.emit(val.tag, val.value, val.currency)

    def on_ib_error(self, reqId, errorCode, errorString, contract=None):
        if 2100 <= errorCode <= 2110 or errorCode == 2158:
            self.log(f"IB Info: {errorString}")
            return
        msg = f"IB Error {errorCode} (Req {reqId}): {errorString}"
        if contract: msg += f" | Contract: {getattr(contract, 'symbol', '')}"
        self.log(msg)

    def on_order_status(self, trade):
        self.log(f"Order status: {trade.orderStatus.status} id={trade.order.orderId}")

    def on_exec_details(self, trade, fill):
        self.log(f"Exec details: {fill.execution.side} {fill.execution.shares} {fill.contract.symbol} @ {fill.execution.price}")

    def _on_bar_update(self, bars, hasNewBar):
        if not hasNewBar: return
        try:
            bar = bars[-1]
            symbol = bars.contract.symbol
            if symbol in self.data:
                df = self.data[symbol]
                new_row = pd.DataFrame([{
                    "datetime": pd.to_datetime(bar.time),
                    "open": bar.open, "high": bar.high, "low": bar.low, "close": bar.close
                }]).set_index("datetime")

                # Check for duplicate index before appending
                if new_row.index[0] not in df.index:
                    self.data[symbol] = pd.concat([df, new_row])
                    self.log(f"Live bar for {symbol} added. Total bars: {len(self.data[symbol])}")
                else:
                    # Update the last bar if timestamp is the same
                    df.loc[new_row.index[0]] = new_row.iloc[0]
                    self.log(f"Live bar for {symbol} updated.")
        except Exception as e:
            self.log(f"_on_bar_update error: {e}")

    def on_add_instrument(self):
        s = self.symbol_entry.text().strip().upper()
        sec = self.sec_type.currentText().strip().upper()
        if not s:
            QMessageBox.warning(self, "Input", "Enter a symbol.")
            return

        dtype = 1 if sec == "CASH" else 3 # Live for Forex, Delayed for others
        self.instruments[s] = {"symbol": s, "secType": sec, "dataType": dtype}
        dtype_name = "Live" if dtype == 1 else "Delayed"
        self.log(f"Instrument added: {s} ({sec}), dataType={dtype_name}")
        self.instrument_list_update_signal.emit(list(self.instruments.keys()))

    def on_remove_selected_instruments(self):
        selected_rows = sorted(set(idx.row() for idx in self.inst_table.selectedIndexes()), reverse=True)
        removed = []
        for r in selected_rows:
            item = self.inst_table.item(r, 0)
            if item:
                sym = item.text()
                removed.append(sym)
                if sym in self.instruments: del self.instruments[sym]
                if sym in self.live_bar_subscriptions:
                    self.ib.cancelMktData(self.live_bar_subscriptions[sym])
                    del self.live_bar_subscriptions[sym]
                self.inst_table.removeRow(r)
        self.instrument_list_update_signal.emit(list(self.instruments.keys()))
        if removed: self.log(f"Removed instruments: {', '.join(removed)}")

    async def _fetch_and_store_historical(self, symbol: str, duration="2 D", bar_size="5 mins"):
        try:
            if not self.ib or not self.ib.isConnected():
                if default_rng is None:
                    self.log("NumPy not available for fake data.")
                    return None
                rng = default_rng()
                n = 200
                times = pd.to_datetime(pd.date_range(end=pd.Timestamp.now(), periods=n, freq='5T'))
                df = pd.DataFrame(index=times)
                df['open'] = 100 + (np.cumsum(rng.standard_normal(n)) * 0.05)
                df['high'] = df['open'] + (abs(rng.standard_normal(n)) * 0.2)
                df['low'] = df['open'] - (abs(rng.standard_normal(n)) * 0.2)
                df['close'] = df['open'] + (rng.standard_normal(n) * 0.03)
                df.reset_index(inplace=True)
                df.rename(columns={'index': 'datetime'}, inplace=True)
                df.set_index('datetime', inplace=True)
                self.data[symbol] = df
                self.log(f"Candlestick data (fake) for {symbol} generated. Last bar: {df.index[-1]}")
                return df

            contract_info = self.instruments.get(symbol)
            if not contract_info:
                self.log(f"No contract for {symbol}")
                return None

            sec_type = contract_info.get("secType", "STK").upper()
            if sec_type == "STK": ib_contract = Stock(symbol, 'SMART', 'USD')
            elif sec_type == "CASH": ib_contract = Forex(symbol)
            else:
                self.log(f"Unsupported secType for historical data: {sec_type}")
                return None

            what_to_show = 'MIDPOINT' if sec_type == 'CASH' else 'TRADES'
            use_rth = sec_type != 'CASH'

            bars = await self.ib.reqHistoricalDataAsync(
                ib_contract, endDateTime='', durationStr=duration,
                barSizeSetting=bar_size, whatToShow=what_to_show, useRTH=use_rth
            )

            if not bars:
                self.log(f"No historical bars for {symbol}")
                return None

            df = util.df(bars)
            df.rename(columns={'date': 'datetime'}, inplace=True)
            df.set_index('datetime', inplace=True)
            self.data[symbol] = df
            self.log(f"Historical data for {symbol} loaded. Last bar: {df.index[-1]}")
            return df
        except Exception as e:
            self.log(f"Historical fetch error for {symbol}: {e}")
            return None

    def on_select_all_toggled(self, state):
        checked = state == getattr(Qt, "Checked", Qt.CheckState.Checked)
        for cb in self.strategy_checkboxes.values():
            cb.blockSignals(True)
            cb.setChecked(checked)
            cb.blockSignals(False)

    def _on_individual_strategy_toggled(self, state):
        all_checked = all(cb.isChecked() for cb in self.strategy_checkboxes.values())
        self.select_all_chk.blockSignals(True)
        self.select_all_chk.setChecked(all_checked)
        self.select_all_chk.blockSignals(False)

    def on_create_checked_strategies(self):
        self.add_task(self._create_checked_strategies_async())

    async def _create_checked_strategies_async(self):
        checked = [name for name, cb in self.strategy_checkboxes.items() if cb.isChecked()]
        if not checked:
            QMessageBox.information(self, "No strategies", "Select at least one strategy.")
            return

        params = {
            "qty": float(self.qty_spin.value()),
            "interval_seconds": float(self.interval_spin.value()),
            "auto_trade": bool(self.auto_trade_chk.isChecked()),
            "wma_fast": self.wma_fast_spin.value(),
            "rma_slow": self.rma_slow_spin.value(),
            "macd_fast": self.macd_fast_spin.value(),
            "macd_slow": self.macd_slow_spin.value(),
            "macd_signal": self.macd_signal_spin.value(),
            "fvg_min_gap_size": self.fvg_min_gap_spin.value(),
            "fvg_trail_atr_mult": self.fvg_trail_atr_mult_spin.value(),
            "fvg_dynamic_sizing": self.fvg_dynamic_sizing_chk.isChecked(),
            "fvg_partial_tp": self.fvg_partial_tp_chk.isChecked(),
            "fvg_trailing_stop": self.fvg_trailing_stop_chk.isChecked(),
        }

        apply_to_selected = self.apply_to_selected_chk.isChecked()
        symbol_list = []
        if apply_to_selected:
            rows = sorted(set(idx.row() for idx in self.inst_table.selectedIndexes()))
            if not rows:
                QMessageBox.information(self, "Select", "Select instrument rows or disable 'Apply to selected'.")
                return
            for r in rows:
                it = self.inst_table.item(r, 0)
                if it: symbol_list.append(it.text())
        else:
            sym = self.single_symbol_dropdown.currentText().strip()
            if not sym:
                QMessageBox.information(self, "Select", "Choose a symbol.")
                return
            symbol_list = [sym]

        for sym in symbol_list:
            for st in checked:
                await self._create_and_start_strategy(sym, st, params)

    async def _subscribe_to_instrument_data(self, symbol: str):
        if symbol not in self.data:
            self.log(f"Fetching initial historical data for {symbol}...")
            await self._fetch_and_store_historical(symbol)

        if self.ib and self.ib.isConnected() and symbol not in self.live_bar_subscriptions:
            self.log(f"Requesting live bars for {symbol}...")
            await self._request_live_bars(symbol)

    async def _request_live_bars(self, symbol: str):
        contract_info = self.instruments.get(symbol)
        if not contract_info or not self.ib: return
        sec_type = contract_info.get("secType", "STK").upper()

        if sec_type == "STK": contract = Stock(symbol, 'SMART', 'USD')
        elif sec_type == "CASH": contract = Forex(symbol)
        else:
            self.log(f"Live bars not supported for secType: {sec_type}")
            return

        try:
            # The fifth argument (False) means we don't want regular TWS updates, just our 5-second bars
            live_bars = await self.ib.reqRealTimeBarsAsync(contract, 5, 'TRADES' if sec_type == 'STK' else 'MIDPOINT', False, [])
            self.live_bar_subscriptions[symbol] = live_bars
            self.log(f"Successfully subscribed to live bars for {symbol}")
        except Exception as e:
            self.log(f"reqRealTimeBarsAsync failed for {symbol}: {e}")

    async def _create_and_start_strategy(self, symbol: str, strat_type: str, params: Dict[str, Any]):
        await self._subscribe_to_instrument_data(symbol)

        strategy = None
        if strat_type == "WMA_RMA_MACD": strategy = WmaRmaMacdStrategy(self, symbol, params)
        elif strat_type == "FairValueGap": strategy = FairValueGapStrategy(self, symbol, params)
        else:
            self.log(f"Strategy {strat_type} not implemented for live trading yet.")
            return

        sid = self.next_strategy_id
        self.next_strategy_id += 1
        self.strategies[sid] = strategy

        row = self.running_strats_table.rowCount()
        self.running_strats_table.insertRow(row)
        self.running_strats_table.setItem(row, 0, QTableWidgetItem(str(sid)))
        self.running_strats_table.setItem(row, 1, QTableWidgetItem(symbol))
        self.running_strats_table.setItem(row, 2, QTableWidgetItem(strat_type))
        self.running_strats_table.setItem(row, 3, QTableWidgetItem(str(params.get("qty", 1))))
        self.running_strats_table.setItem(row, 4, QTableWidgetItem(str(params.get("interval_seconds", 5.0))))
        stop_btn = QPushButton("Stop")
        stop_btn.clicked.connect(lambda sid_local=sid: self.stop_strategy(sid_local))
        self.running_strats_table.setCellWidget(row, 5, stop_btn)

        task = self.add_task(strategy.run())
        if task:
            self.strategy_tasks[sid] = task
            self.log(f"Strategy {sid} started: {strat_type} on {symbol}")

    def stop_strategy(self, sid: int):
        strategy = self.strategies.get(sid)
        if strategy: strategy.stop()
        task = self.strategy_tasks.get(sid)
        if task: task.cancel()

        for r in range(self.running_strats_table.rowCount()):
            it = self.running_strats_table.item(r, 0)
            if it and it.text() == str(sid):
                self.running_strats_table.removeRow(r)
                break
        self.strategies.pop(sid, None)
        self.strategy_tasks.pop(sid, None)
        self.log(f"Strategy {sid} stopped.")

    def save_config(self):
        try:
            with open(CONFIG_FILE, "w") as f:
                json.dump({"instruments": self.instruments}, f)
            self.log("Config saved.")
        except Exception as e:
            self.log(f"save_config error: {e}")

    def load_config(self):
        if not os.path.exists(CONFIG_FILE): return
        try:
            with open(CONFIG_FILE, "r") as f:
                conf = json.load(f)
            self.instruments = conf.get("instruments", {})
            self.instrument_list_update_signal.emit(list(self.instruments.keys()))
            self.log("Config loaded.")
        except Exception as e:
            self.log(f"load_config error: {e}")

    def update_instrument_selectors(self, names: List[str]):
        self.single_symbol_dropdown.clear()
        self.inst_table.setRowCount(0)
        for s in names:
            self.single_symbol_dropdown.addItem(s)
            info = self.instruments.get(s, {})
            row = self.inst_table.rowCount()
            self.inst_table.insertRow(row)
            self.inst_table.setItem(row, 0, QTableWidgetItem(s))
            self.inst_table.setItem(row, 1, QTableWidgetItem(info.get("secType", "")))
            dtype = info.get("dataType", 3)
            dtype_name = "Live" if dtype == 1 else "Delayed"
            self.inst_table.setItem(row, 2, QTableWidgetItem(dtype_name))

    def update_summary_text(self, text: str):
        self.summary_text.setText(text)

    def update_account_table(self, tag, value, currency):
        for r in range(self.account_table.rowCount()):
            it = self.account_table.item(r, 0)
            if it and it.text() == tag:
                self.account_table.item(r, 1).setText(str(value))
                self.account_table.item(r, 2).setText(str(currency))
                return
        r = self.account_table.rowCount()
        self.account_table.insertRow(r)
        self.account_table.setItem(r, 0, QTableWidgetItem(tag))
        self.account_table.setItem(r, 1, QTableWidgetItem(str(value)))
        self.account_table.setItem(r, 2, QTableWidgetItem(str(currency)))

    def update_positions_table(self, sym, qty, avg):
        for r in range(self.positions_table.rowCount()):
            it = self.positions_table.item(r, 0)
            if it and it.text() == sym:
                if qty == 0: self.positions_table.removeRow(r)
                else:
                    self.positions_table.item(r, 1).setText(str(qty))
                    self.positions_table.item(r, 2).setText(str(avg))
                return
        if qty != 0:
            r = self.positions_table.rowCount()
            self.positions_table.insertRow(r)
            self.positions_table.setItem(r, 0, QTableWidgetItem(sym))
            self.positions_table.setItem(r, 1, QTableWidgetItem(str(qty)))
            self.positions_table.setItem(r, 2, QTableWidgetItem(str(avg)))

class BaseStrategy:
    def __init__(self, app: TradingApp, symbol: str, params: Optional[Dict[str, Any]] = None):
        self.app = app
        self.symbol = symbol
        self.params = params or {}
        self.auto_trade = bool(self.params.get("auto_trade", False))
        self.interval = float(self.params.get("interval_seconds", 5.0))
        self.is_running = False
        self.contract = self.app.instruments.get(symbol)
        self.trade_info = {}

    def log(self, msg: str):
        self.app.log(f"[{self.__class__.__name__}:{self.symbol}] {msg}")

    async def place_order(self, action: str, qty: float, order_type: str = "MKT", limit_price: Optional[float] = None, stop_price: Optional[float] = None, order_id: Optional[int] = None):
        if not self.app.ib or not self.app.ib.isConnected():
            self.log("IB not available — skipping order.")
            return None
        if not self.contract:
            self.log("No contract for order.")
            return None

        sec_type = self.contract.get("secType", "STK").upper()
        if sec_type == "STK": ib_contract = Stock(self.symbol, 'SMART', 'USD')
        elif sec_type == "CASH": ib_contract = Forex(self.symbol)
        else:
            self.log(f"Unsupported secType for order: {sec_type}")
            return None

        if order_type.upper() == "MKT":
            order = MarketOrder(action, float(qty))
        elif order_type.upper() == "LMT":
            order = LimitOrder(action, float(qty), float(limit_price) if limit_price else 0.0)
        elif order_type.upper() == "STP":
            order = StopOrder(action, float(qty), float(stop_price) if stop_price else 0.0)
        else:
            self.log(f"Unsupported order type: {order_type}")
            return None

        if order_id:
            order.orderId = order_id

        self.log(f"Placing order: {action} {qty} {self.symbol} @ {order_type}")
        trade = self.app.ib.placeOrder(ib_contract, order)
        self.log(f"Trade object: {trade}")
        return trade

    def get_dataframe(self) -> Optional[pd.DataFrame]:
        df = self.app.data.get(self.symbol)
        if df is None or df.empty: return None
        return df.copy()

    async def run(self):
        self.is_running = True
        self.log("Base strategy started (does nothing).")
        try:
            while self.is_running:
                await asyncio.sleep(self.interval)
        except asyncio.CancelledError:
            self.is_running = False
        finally:
            self.log("Base strategy stopped.")

    def stop(self):
        self.is_running = False

class WmaRmaMacdStrategy(BaseStrategy):
    def __init__(self, app, symbol, params=None):
        super().__init__(app, symbol, params)
        self.wma_fast = int(self.params.get("wma_fast", 50))
        self.rma_slow = int(self.params.get("rma_slow", 100))
        self.macd_fast = int(self.params.get("macd_fast", 5))
        self.macd_slow = int(self.params.get("macd_slow", 10))
        self.macd_signal = int(self.params.get("macd_signal", 10))
        self.min_bars = max(self.wma_fast, self.rma_slow) + 5

    def calculate_indicators(self, df: pd.DataFrame):
        df.ta.wma(length=self.wma_fast, append=True, col_names=(f'WMA_{self.wma_fast}',))
        df.ta.rma(length=self.rma_slow, append=True, col_names=(f'RMA_{self.rma_slow}',))
        df.ta.macd(fast=self.macd_fast, slow=self.macd_slow, signal=self.macd_signal, append=True)
        normalize_ta_columns(df)

    def check_conditions(self, df: pd.DataFrame) -> Optional[str]:
        if len(df) < self.min_bars: return None
        price = df['close'].iloc[-1]
        wf_col = f"WMA_{self.wma_fast}"
        rs_col = f"RMA_{self.rma_slow}"
        if wf_col not in df.columns or rs_col not in df.columns or "MACD_hist" not in df.columns:
            return None

        wv = df[wf_col].iloc[-1]
        sv = df[rs_col].iloc[-1]
        macd_hist = df["MACD_hist"].iloc[-1]
        prev_macd_hist = df["MACD_hist"].iloc[-2]

        if price > wv > sv and macd_hist > 0 and macd_hist > prev_macd_hist: return "BUY"
        if price < wv < sv and macd_hist < 0 and macd_hist < prev_macd_hist: return "SELL"
        return None

    async def run(self):
        self.is_running = True
        self.log("WMA/RMA+MACD strategy started.")
        try:
            while self.is_running:
                await asyncio.sleep(self.interval)
                df = self.get_dataframe()
                if df is None: continue
                self.calculate_indicators(df)
                sig = self.check_conditions(df)
                if sig:
                    self.log(f"Signal detected: {sig}")
                    if self.auto_trade:
                        await self.place_order(sig, float(self.params.get("qty", 1)))
        except asyncio.CancelledError:
            self.log("WMA/RMA+MACD task cancelled.")
        finally:
            self.is_running = False
            self.log("WMA/RMA+MACD strategy stopped.")

class FairValueGapStrategy(BaseStrategy):
    def __init__(self, app, symbol, params=None):
        super().__init__(app, symbol, params)
        self.qty = float(self.params.get("qty", 1.0))
        self.min_gap_size = float(self.params.get("fvg_min_gap_size", 0.1))
        self.trail_atr_mult = float(self.params.get("fvg_trail_atr_mult", 1.0))
        self.use_dynamic_sizing = bool(self.params.get("fvg_dynamic_sizing", True))
        self.use_partial_tp = bool(self.params.get("fvg_partial_tp", True))
        self.use_trailing_stop = bool(self.params.get("fvg_trailing_stop", True))
        self.active_fvgs = []
        self.last_processed_bar_time = None

    def _calculate_indicators(self, df):
        df.ta.atr(length=14, append=True)
        df.ta.adx(length=14, append=True)
        df.ta.sma(length=50, append=True, col_names=('SMA_50',))
        normalize_ta_columns(df)

    def _is_trending(self, df):
        if 'ADX' not in df.columns: return False
        return df['ADX'].iloc[-1] > 25

    def _is_break_of_structure(self, df, current_idx, fvg_low, direction="bullish"):
        lookback = 100
        start_idx = max(0, current_idx - lookback)
        recent_df = df.iloc[start_idx:current_idx]
        if direction == "bullish":
            return df['close'].iloc[current_idx] >= recent_df['high'].max()
        else: # Bearish
            return df['close'].iloc[current_idx] <= recent_df['low'].min()

    def _detect_new_fvg(self, df, i):
        if i < 2: return
        high_i2, low_i = df['high'].iloc[i-2], df['low'].iloc[i]
        if high_i2 < low_i and (low_i - high_i2) >= self.min_gap_size:
            if self._is_trending(df) or self._is_break_of_structure(df, i, high_i2, "bullish"):
                self.active_fvgs.append({"start_idx": i, "low": high_i2, "high": low_i, "direction": "bullish", "tested": False})

        low_i2, high_i = df['low'].iloc[i-2], df['high'].iloc[i]
        if low_i2 > high_i and (low_i2 - high_i) >= self.min_gap_size:
            if self._is_trending(df) or self._is_break_of_structure(df, i, low_i2, "bearish"):
                self.active_fvgs.append({"start_idx": i, "low": high_i, "high": low_i2, "direction": "bearish", "tested": False})

    def _update_active_fvgs(self, df, i):
        current_candle = df.iloc[i]
        signal = None
        to_remove = []
        for idx, fvg in enumerate(self.active_fvgs):
            if not fvg['tested']:
                if (fvg['direction'] == "bullish" and current_candle['low'] <= fvg['high']) or \
                   (fvg['direction'] == "bearish" and current_candle['high'] >= fvg['low']):
                    fvg['tested'] = True
            else:
                if (fvg['direction'] == "bullish" and current_candle['close'] > fvg['high']) or \
                   (fvg['direction'] == "bearish" and current_candle['close'] < fvg['low']):
                    signal = "BUY" if fvg['direction'] == "bullish" else "SELL"
                    self.trade_info['fvg_low'] = fvg['low']
                    self.trade_info['fvg_high'] = fvg['high']
                    to_remove.append(idx)
                elif i - fvg['start_idx'] > 10: # Expire after 10 candles
                    to_remove.append(idx)

        for r_idx in sorted(to_remove, reverse=True): self.active_fvgs.pop(r_idx)
        return signal

    async def _manage_open_position(self, df):
        if not self.trade_info or not self.use_trailing_stop: return

        current_price = df['close'].iloc[-1]
        atr = df['ATR_14'].iloc[-1]

        if self.trade_info['direction'] == 'BUY':
            if self.use_partial_tp and not self.trade_info.get('partial_taken') and current_price >= self.trade_info['partial_tp_price']:
                await self.place_order("SELL", self.qty / 2, order_type="MKT")
                self.trade_info['partial_taken'] = True
                self.log("Partial take profit hit for BUY.")

            new_sl = current_price - self.trail_atr_mult * atr
            if new_sl > self.trade_info['sl_price']:
                self.trade_info['sl_price'] = new_sl
                await self.place_order("SELL", self.qty / 2 if self.trade_info.get('partial_taken') else self.qty, order_type="STP", stop_price=new_sl, order_id=self.trade_info['stop_order_id'])
                self.log(f"Trailing stop for BUY moved to {new_sl:.2f}")

        elif self.trade_info['direction'] == 'SELL':
            if self.use_partial_tp and not self.trade_info.get('partial_taken') and current_price <= self.trade_info['partial_tp_price']:
                await self.place_order("BUY", self.qty / 2, order_type="MKT")
                self.trade_info['partial_taken'] = True
                self.log("Partial take profit hit for SELL.")

            new_sl = current_price + self.trail_atr_mult * atr
            if new_sl < self.trade_info['sl_price']:
                self.trade_info['sl_price'] = new_sl
                await self.place_order("BUY", self.qty / 2 if self.trade_info.get('partial_taken') else self.qty, order_type="STP", stop_price=new_sl, order_id=self.trade_info['stop_order_id'])
                self.log(f"Trailing stop for SELL moved to {new_sl:.2f}")

    async def run(self):
        self.is_running = True
        self.log("FairValueGap strategy started.")
        try:
            while self.is_running:
                await asyncio.sleep(self.interval)
                df = self.get_dataframe()
                if df is None or len(df) < 50: continue

                if self.last_processed_bar_time == df.index[-1]:
                    if self.trade_info: await self._manage_open_position(df)
                    continue
                self.last_processed_bar_time = df.index[-1]

                self._calculate_indicators(df)

                if not self.trade_info:
                    self._detect_new_fvg(df, len(df) - 2)
                    sig = self._update_active_fvgs(df, len(df) - 1)
                    if sig:
                        self.log(f"FVG Signal detected: {sig}")
                        if self.auto_trade:
                            entry_price = df['close'].iloc[-1]
                            atr = df['ATR_14'].iloc[-1]
                            sl_price = entry_price - self.trail_atr_mult * atr if sig == "BUY" else entry_price + self.trail_atr_mult * atr
                            risk = abs(entry_price - sl_price)

                            self.trade_info = {
                                "direction": sig,
                                "entry_price": entry_price,
                                "sl_price": sl_price,
                                "partial_tp_price": entry_price + 1.5 * risk if sig == "BUY" else entry_price - 1.5 * risk,
                            }

                            trade = await self.place_order(sig, self.qty, order_type="MKT")
                            if trade:
                                stop_trade = await self.place_order("SELL" if sig == "BUY" else "BUY", self.qty, order_type="STP", stop_price=sl_price)
                                if stop_trade: self.trade_info['stop_order_id'] = stop_trade.order.orderId
                else:
                    pos = self.app.ib.positions()
                    if not any(p.contract.symbol == self.symbol and p.position != 0 for p in pos):
                        self.trade_info = {} # Reset if position closed
                        self.log("Position closed, resetting trade info.")
                    else:
                        await self._manage_open_position(df)
        except asyncio.CancelledError:
            self.log("FairValueGap task cancelled.")
        finally:
            self.is_running = False
            self.log("FairValueGap strategy stopped.")

if __name__ == "__main__":
    try:
        qt_app = QApplication(sys.argv)
        loop = qasync.QEventLoop(qt_app)
        asyncio.set_event_loop(loop)
        main_win = TradingApp()
        main_win.show()
        with loop:
            loop.run_forever()
    except Exception as e:
        print("Fatal error running the app:", e)
        traceback.print_exc()
